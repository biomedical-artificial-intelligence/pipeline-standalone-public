import os
import subprocess
import logging
import shutil
import tempfile
from multiprocessing import Pool

from config import (
    DISTANCE_BIN,
    DISTANCE_ANTIBODY_BIN,
    DOCK_CREATE_PL,
    DOCK_CREATE_LIG_BIN,
    SCRATCH_DIR,
)

logger = logging.getLogger(__name__)


def calculate_distance_fortran_worker(complex_pdb_path, chain_id, target_res, peptide_length):
    """Fortran distance (peptide variant - residues 1..peptide_length)."""
    try:
        pid = os.getpid()

        with tempfile.TemporaryDirectory(prefix=f"dist_{pid}_") as tmpdir:
            cmd = [
                DISTANCE_BIN,
                complex_pdb_path,
                chain_id,
                str(target_res),
                "1",
                str(peptide_length),
                "P",
            ]

            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            fort_file = os.path.join(tmpdir, "fort.2")
            if os.path.exists(fort_file):
                with open(fort_file, "r") as f:
                    return float(f.read().strip())

            return None

    except Exception as e:
        logger.error(f"Distance calculation error: {e}")
        return None


def calculate_distance_fortran_worker_antibody(complex_pdb_path, chain_id, target_res, cdr_start, cdr_end, cdr_chain_id):
    """Fortran distance (antibody variant - CDR region only)."""
    try:
        pid = os.getpid()

        with tempfile.TemporaryDirectory(prefix=f"dist_{pid}_") as tmpdir:
            cmd = [
                DISTANCE_ANTIBODY_BIN,
                complex_pdb_path,
                chain_id,
                str(target_res),
                str(cdr_start),
                str(cdr_end),
                cdr_chain_id,
            ]

            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            fort_file = os.path.join(tmpdir, "fort.2")
            if os.path.exists(fort_file):
                with open(fort_file, "r") as f:
                    return float(f.read().strip())

            return None

    except Exception as e:
        logger.error(f"Distance calculation error (antibody): {e}")
        return None


def extract_dock_score_fast(dock_output_path, pose_number):
    """Extract the ZDOCK score for a given pose."""
    try:
        with open(dock_output_path, "r") as f:
            pose_count = 0
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split()

                    if len(parts) == 7:
                        try:
                            score = float(parts[6])
                            pose_count += 1
                            if pose_count == pose_number:
                                return score
                        except ValueError:
                            continue
        return None

    except Exception:
        return None


def process_single_pose_fortran(args):
    """Process a single pose (distance + ZDOCK score -> geometric-mean score)."""
    (pose_num, complex_pdb_path, chain_id, target_res, peptide_length,
     distance_threshold, dock_output_path, w1, w2,
     DOCK_MIN, DOCK_MAX, DISTANCE_MAX,
     cdr_start_position, cdr_end_position, cdr_chain_id) = args

    try:
        if not os.path.exists(complex_pdb_path):
            return None

        if cdr_start_position is not None and cdr_end_position is not None:
            distance = calculate_distance_fortran_worker_antibody(
                complex_pdb_path, chain_id, target_res,
                cdr_start_position, cdr_end_position, cdr_chain_id
            )
        else:
            distance = calculate_distance_fortran_worker(
                complex_pdb_path, chain_id, target_res, peptide_length
            )

        if distance is None or distance > distance_threshold:
            return None

        dock_score = extract_dock_score_fast(dock_output_path, pose_num)

        if dock_score is None:
            return None

        dock_norm = (dock_score - DOCK_MIN) / (DOCK_MAX - DOCK_MIN)
        dock_norm = max(0.0, min(1.0, dock_norm))

        proximity_norm = (DISTANCE_MAX - distance) / DISTANCE_MAX
        proximity_norm = max(0.0, min(1.0, proximity_norm))

        if dock_norm <= 0 or proximity_norm <= 0:
            enhanced_score = 0.0
        else:
            enhanced_score = ((dock_norm ** w1) * (proximity_norm ** w2)) ** (1 / (w1 + w2))

        return {
            "pose_number": pose_num,
            "dock_score": dock_score,
            "distance": distance,
            "enhanced_score": enhanced_score,
            "dock_norm": dock_norm,
            "proximity_norm": proximity_norm,
            "pdb_path": complex_pdb_path,
        }

    except Exception:
        return None


def _safe_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass


class GeometricMeanPoseRanker:
    """Fortran distance + Python multiprocessing pose ranking."""

    def __init__(
        self,
        dock_output_path=None,
        receptor_path=None,
        peptide_length=10,
        cdr_start_position=None,
        cdr_end_position=None,
        cdr_chain_id="L",
        n_processes=2,
        distance_max=50.0,
    ):
        self.dock_output_path = dock_output_path
        self.receptor_path = receptor_path
        self.peptide_length = peptide_length
        self.cdr_start_position = cdr_start_position
        self.cdr_end_position = cdr_end_position
        self.cdr_chain_id = cdr_chain_id
        self.DOCK_MIN = 0.5
        self.DOCK_MAX = 3.0
        # Proximity normalization scale: antibody=50.0, peptide=15.0 (injected per mode).
        self.DISTANCE_MAX = distance_max
        self.n_processes = n_processes

    def create_complex_structures_batch(self, target_dir, max_poses, pipeline_id, job_id):
        """Generate complex structures from the ZDOCK output via create.pl."""
        dock_work_dir = None
        try:
            if not os.path.exists(self.dock_output_path):
                logger.error(f"ZDOCK output not found: {self.dock_output_path}")
                return False

            # Per-job working directory under scratch; create.pl needs its
            # helpers (create_lig) and the receptor co-located in cwd.
            target_name = os.path.basename(target_dir)
            dock_work_dir = os.path.join(SCRATCH_DIR, f"{target_name}_{pipeline_id}_{job_id}")
            os.makedirs(dock_work_dir, exist_ok=True)

            _safe_symlink(DOCK_CREATE_PL, os.path.join(dock_work_dir, "create.pl"))
            _safe_symlink(DOCK_CREATE_LIG_BIN, os.path.join(dock_work_dir, "create_lig"))
            _safe_symlink(self.receptor_path, os.path.join(dock_work_dir, "receptor"))

            cmd = ["./create.pl", os.path.abspath(self.dock_output_path), str(max_poses)]

            result = subprocess.run(
                cmd,
                cwd=dock_work_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error("create.pl failed")
                return False

            moved_count = 0
            for pose_num in range(1, max_poses + 1):
                src_file = os.path.join(dock_work_dir, f"complex.{pose_num}.pdb")
                dst_file = os.path.join(target_dir, f"complex.{pose_num}.pdb")

                if os.path.exists(src_file):
                    shutil.move(src_file, dst_file)
                    moved_count += 1

            return moved_count > 0

        except Exception as e:
            logger.error(f"Complex structure creation error: {e}")
            return False

        finally:
            if dock_work_dir and os.path.exists(dock_work_dir):
                shutil.rmtree(dock_work_dir, ignore_errors=True)

    def get_best_enhanced_pose_parallel(self, target_residue, pipeline_id, job_id, max_poses=2000, distance_threshold=50.0, w1=1.0, w2=1.0):
        """Select the best pose via multiprocessing."""
        target_dir = os.path.join(
            SCRATCH_DIR,
            f"{target_residue.replace(':', '_')}_{pipeline_id}_{job_id}_fortran_parallel",
        )
        os.makedirs(target_dir, exist_ok=True)

        success = self.create_complex_structures_batch(target_dir, max_poses, pipeline_id, job_id)
        if not success:
            logger.error("Complex structure creation failed")
            return None, None

        chain_id, target_res = target_residue.split(":")

        process_args = []
        for pose_num in range(1, max_poses + 1):
            complex_pdb_path = os.path.join(target_dir, f"complex.{pose_num}.pdb")
            process_args.append((
                pose_num, complex_pdb_path, chain_id, int(target_res), self.peptide_length,
                distance_threshold, self.dock_output_path, w1, w2,
                self.DOCK_MIN, self.DOCK_MAX, self.DISTANCE_MAX,
                self.cdr_start_position, self.cdr_end_position, self.cdr_chain_id,
            ))

        with Pool(processes=self.n_processes) as pool:
            results = pool.map(process_single_pose_fortran, process_args)

        valid_poses = [r for r in results if r is not None]

        if not valid_poses:
            return None, None

        best_pose = max(valid_poses, key=lambda x: x["enhanced_score"])

        return best_pose, valid_poses
