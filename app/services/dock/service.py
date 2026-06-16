import os
import shutil
import asyncio
import logging
import subprocess
import tempfile
from typing import Dict, Any

from config import ZDOCK_DIR, ZDOCK_BIN, MARK_SUR_BIN, SCRATCH_DIR

logger = logging.getLogger(__name__)

MAX_POSES = 2000


def _zdock_env():
    """Environment for ZDOCK toolkit subprocesses.

    Prepends ZDOCK_DIR to LD_LIBRARY_PATH so shared libraries shipped next to
    the binaries are found at runtime — notably ``libg2c.so.0`` (the legacy g77
    runtime) that ``mark_sur`` is linked against and that modern distros no
    longer provide.
    """
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ZDOCK_DIR + (os.pathsep + existing if existing else "")
    return env


class DockService:
    """ZDOCK execution and pose-filtering service (antibody CDR)."""

    def __init__(self, executor=None, n_processes=2):
        self.iter_dir = ""
        # Dedicated executor for ZDOCK subprocess / filtering offload.
        # None falls back to the loop's default ThreadPoolExecutor(min(32, cpu+4)).
        self.executor = executor
        # Worker processes for the distance/pose-ranking multiprocessing pool.
        self.n_processes = n_processes

    async def run_single_dock(
        self,
        job_id: int,
        iter_dir: str,
        receptor_pdb: str,
        target_residue: str,
        pipeline_id: int,
        cdr_start_position: int = None,
        cdr_end_position: int = None,
        cdr_chain_id: str = "L",
        peptide_length: int = 10,
        distance_threshold: float = 50.0,
    ) -> Dict[str, Any]:

        loop = asyncio.get_event_loop()
        temp_dir = tempfile.mkdtemp(prefix=f"marksur_{pipeline_id}_{job_id}_")

        try:
            receptor_marksur = os.path.join(temp_dir, f"marksur_{os.path.basename(receptor_pdb)}")
            ligand_pdb = os.path.join(iter_dir, "input_batch_01.pdb")
            ligand_marksur = os.path.join(temp_dir, "marksur_ligand_01.pdb")
            zdock_out = os.path.join(iter_dir, "zdock_batch_01.out")

            await loop.run_in_executor(self.executor, lambda: subprocess.run(
                [MARK_SUR_BIN, receptor_pdb, receptor_marksur],
                cwd=ZDOCK_DIR,
                env=_zdock_env(),
                check=True,
                capture_output=True,
            ))

            await loop.run_in_executor(self.executor, lambda: subprocess.run(
                [MARK_SUR_BIN, ligand_pdb, ligand_marksur],
                cwd=ZDOCK_DIR,
                env=_zdock_env(),
                check=True,
                capture_output=True,
            ))

            await loop.run_in_executor(self.executor, lambda: subprocess.run(
                [
                    ZDOCK_BIN,
                    "-R", receptor_marksur,
                    "-L", ligand_marksur,
                    "-o", zdock_out,
                ],
                cwd=ZDOCK_DIR,
                env=_zdock_env(),
                capture_output=True,
            ))

            if not os.path.exists(zdock_out) or os.path.getsize(zdock_out) == 0:
                raise Exception(f"ZDOCK output not found: {zdock_out}")

            result = await self._run_filtering_for_batch(
                batch_str="01",
                zdock_output_path=zdock_out,
                receptor_path=receptor_pdb,
                ligand_path=ligand_pdb,
                target_residue=target_residue,
                cdr_start_position=cdr_start_position,
                cdr_end_position=cdr_end_position,
                cdr_chain_id=cdr_chain_id,
                peptide_length=peptide_length,
                distance_threshold=distance_threshold,
                pipeline_id=pipeline_id,
                job_id=job_id,
                iter_dir=iter_dir,
            )

            logger.info(f"ZDOCK complete - job {job_id}: score={result.get('score', 0):.4f}")
            return result

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def _run_filtering_for_batch(
        self,
        batch_str: str,
        zdock_output_path: str,
        receptor_path: str,
        ligand_path: str,
        target_residue: str,
        cdr_start_position: int,
        cdr_end_position: int,
        cdr_chain_id: str,
        peptide_length: int,
        distance_threshold: float = 50.0,
        pipeline_id: int = 0,
        job_id: int = 0,
        iter_dir: str = "",
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()

        target_dir = os.path.join(
            SCRATCH_DIR,
            f"{target_residue.replace(':', '_')}_{pipeline_id}_{job_id}_fortran_parallel",
        )

        try:
            from .filtering import GeometricMeanPoseRanker

            if not os.path.exists(zdock_output_path):
                logger.error(f"ZDOCK output not found - Batch {batch_str}")
                return {"score": 0.0, "ranking": None}

            ranker = GeometricMeanPoseRanker(
                zdock_output_path=zdock_output_path,
                receptor_path=receptor_path,
                peptide_length=peptide_length,
                cdr_start_position=cdr_start_position,
                cdr_end_position=cdr_end_position,
                cdr_chain_id=cdr_chain_id,
                n_processes=self.n_processes,
                distance_max=distance_threshold,
            )

            best_pose, valid_poses = await loop.run_in_executor(
                self.executor,
                lambda: ranker.get_best_enhanced_pose_parallel(
                    target_residue=target_residue,
                    pipeline_id=pipeline_id,
                    job_id=job_id,
                    max_poses=MAX_POSES,
                    distance_threshold=distance_threshold,
                    w1=1.0,
                    w2=1.0,
                )
            )

            if best_pose is None or not valid_poses:
                return {"score": 0.0, "ranking": None}

            best_pose = max(valid_poses, key=lambda x: x["enhanced_score"])

            if best_pose and "pdb_path" in best_pose:
                complex_pdb_path = os.path.join(iter_dir, f"complex_batch_{batch_str}.pdb")
                # Convert via a subprocess pymol invocation rather than an
                # in-process import: pymol holds global state that is unsafe
                # across the many concurrent docking jobs run by the executor.
                result = subprocess.run(
                    [
                        "python3", "-c",
                        f"from pymol import cmd; cmd.quiet=1; "
                        f"cmd.load('{best_pose['pdb_path']}', 'complex'); "
                        f"cmd.save('{complex_pdb_path}', 'complex'); cmd.delete('all')"
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.warning(f"PyMOL conversion failed, copying original: {result.stderr}")
                    shutil.copy(best_pose["pdb_path"], complex_pdb_path)

            return {
                "score": best_pose["zdock_score"] / 1000,
                "ranking": best_pose["pose_number"],
            }

        except Exception as e:
            logger.error(f"Filtering error - Batch {batch_str}: {e}")
            return {"score": 0.0, "ranking": None}

        finally:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
