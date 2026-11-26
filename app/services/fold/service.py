import asyncio
import logging
import os
import shutil

from .fold_runner import FoldRunner

logger = logging.getLogger(__name__)


class FoldService:
    """AlphaFold3 fold service (direct subprocess).

    Antibody mode: 2-chain (fixed + modified) via ``run_single_fold``.
    Peptide  mode: 1-chain via ``run_single_fold_peptide``.
    """

    def __init__(self):
        self._runner = FoldRunner(logger_name="FoldService")

    async def run_single_fold(
        self,
        job_id: int,
        fixed_chain_sequence: str,
        modified_chain_sequence: str,
        cdr_chain_id: str,
        pipeline_id: int,
        iter_dir: str,
        gpu_id: int = 0,
    ) -> dict:
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self._runner.run_antibody(
                pipeline_id=pipeline_id,
                fixed_chain_sequence=fixed_chain_sequence,
                modified_chain_sequence=modified_chain_sequence,
                cdr_chain_id=cdr_chain_id,
                iter=job_id,
                gpu_id=gpu_id,
                batch=1,
                max_retries=3,
            ),
        )

        if not result["success"]:
            raise Exception(
                f"AlphaFold3 failed - job {job_id}: {result.get('error_message')}"
            )

        src_pdb = result["pdb_file_path"]
        dst_pdb = os.path.join(iter_dir, "input_batch_01.pdb")
        shutil.copy2(src_pdb, dst_pdb)

        if not os.path.exists(dst_pdb) or os.path.getsize(dst_pdb) == 0:
            raise Exception(f"PDB copy failed - job {job_id}: {dst_pdb}")

        logger.info(f"AlphaFold3 complete - job {job_id}, path={dst_pdb}")
        return result

    async def run_single_fold_peptide(
        self,
        job_id: int,
        sequence: str,
        pipeline_id: int,
        iter_dir: str,
        gpu_id: int = 0,
    ) -> dict:
        """Single-chain (peptide) fold. Writes input_batch_01.pdb to iter_dir."""
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self._runner.run(
                pipeline_id=pipeline_id,
                sequence=sequence,
                iter=job_id,
                gpu_id=gpu_id,
                batch=1,
                max_retries=3,
            ),
        )

        if not result["success"]:
            raise Exception(
                f"AlphaFold3 failed - job {job_id}: {result.get('error_message')}"
            )

        src_pdb = result["pdb_file_path"]
        dst_pdb = os.path.join(iter_dir, "input_batch_01.pdb")
        shutil.copy2(src_pdb, dst_pdb)

        if not os.path.exists(dst_pdb) or os.path.getsize(dst_pdb) == 0:
            raise Exception(f"PDB copy failed - job {job_id}: {dst_pdb}")

        logger.info(f"AlphaFold3 complete (peptide) - job {job_id}, path={dst_pdb}")
        return result
