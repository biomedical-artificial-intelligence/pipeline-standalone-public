import os
import json
import subprocess
import shutil
import time
import logging


class AlphaFold3ParallelRunner:
    """
    AlphaFold3 직접 실행 runner.
    Docker-in-Docker 없이 컨테이너 내부에서 run_alphafold.py를 직접 subprocess로 호출.
    """

    def __init__(self, logger_name=None):
        self.logger = logging.getLogger(logger_name or "AlphaFold3ParallelRunner")

        from config import (
            INPUT_BASE_DIR, OUTPUT_BASE_DIR,
            ALPHAFOLD_PYTHON, ALPHAFOLD_SCRIPT, ALPHAFOLD_MODEL_DIR, ALPHAFOLD_DB_DIR,
        )

        self.input_base_dir = INPUT_BASE_DIR
        self.output_base_dir = OUTPUT_BASE_DIR

        # Interpreter of the user's external AlphaFold3 environment.
        self.alphafold_python = ALPHAFOLD_PYTHON
        self.alphafold_script = ALPHAFOLD_SCRIPT
        self.model_dir = ALPHAFOLD_MODEL_DIR
        self.db_dir = ALPHAFOLD_DB_DIR

        os.makedirs(self.input_base_dir, exist_ok=True)
        os.makedirs(self.output_base_dir, exist_ok=True)

    def _input_dir(self, pipeline_id: int) -> str:
        path = os.path.join(self.input_base_dir, str(pipeline_id))
        os.makedirs(path, exist_ok=True)
        return path

    def _output_dir(self, pipeline_id: int, job_name: str) -> str:
        path = os.path.join(self.output_base_dir, str(pipeline_id), job_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _pipeline_output_dir(self, pipeline_id: int) -> str:
        path = os.path.join(self.output_base_dir, str(pipeline_id))
        os.makedirs(path, exist_ok=True)
        return path

    def create_alphafold_input(self, sequence, job_name, pipeline_id):
        try:
            input_json = {
                "name": job_name,
                "sequences": [
                    {
                        "protein": {
                            "id": ["P"],
                            "sequence": sequence,
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": []
                        }
                    }
                ],
                "modelSeeds": [1],
                "dialect": "alphafold3",
                "version": 1
            }

            input_path = os.path.join(self._input_dir(pipeline_id), f"alphafold_input_{job_name}.json")
            with open(input_path, 'w') as f:
                json.dump(input_json, f, indent=2)

            return {'success': True, 'input_path': input_path}

        except Exception as e:
            self.logger.error(f"Input file creation failed: {e}")
            return {'success': False, 'error_code': 'INPUT_CREATION_FAILED', 'error_message': str(e)}

    def create_alphafold_input_antibody(self, fixed_chain_sequence, modified_chain_sequence,
                                        cdr_chain_id, job_name, pipeline_id):
        """
        Build a 2-chain AlphaFold3 input (antibody).

        The CDR-bearing (modified) chain is labelled `cdr_chain_id` ("L" or "H")
        so the downstream distance_antibody calculation can locate it; the fixed
        chain takes the other label.
        """
        try:
            fixed_chain_id = "H" if cdr_chain_id == "L" else "L"
            input_json = {
                "name": job_name,
                "sequences": [
                    {
                        "protein": {
                            "id": [fixed_chain_id],
                            "sequence": fixed_chain_sequence,
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": []
                        }
                    },
                    {
                        "protein": {
                            "id": [cdr_chain_id],
                            "sequence": modified_chain_sequence,
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": []
                        }
                    }
                ],
                "modelSeeds": [1],
                "dialect": "alphafold3",
                "version": 1
            }

            input_path = os.path.join(self._input_dir(pipeline_id), f"alphafold_input_{job_name}.json")
            with open(input_path, 'w') as f:
                json.dump(input_json, f, indent=2)

            return {'success': True, 'input_path': input_path}

        except Exception as e:
            self.logger.error(f"Antibody input file creation failed: {e}")
            return {'success': False, 'error_code': 'INPUT_CREATION_FAILED', 'error_message': str(e)}

    def run_alphafold(self, job_name, input_path, gpu_id, pipeline_id):
        try:
            output_dir = self._output_dir(pipeline_id, job_name)
            pipeline_output_dir = self._pipeline_output_dir(pipeline_id)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            result = subprocess.run(
                [
                    self.alphafold_python, self.alphafold_script,
                    f"--json_path={input_path}",
                    f"--model_dir={self.model_dir}",
                    f"--db_dir={self.db_dir}",
                    f"--output_dir={pipeline_output_dir}",
                    "--run_inference=true",
                ],
                env=env,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                log_output = result.stdout + result.stderr
                if any(k in log_output.lower() for k in ['out of memory', 'oom', 'cuda error', 'memory error']):
                    error_code = 'GPU_MEMORY_INSUFFICIENT'
                else:
                    error_code = 'EXECUTION_FAILED'
                self.logger.error(f"AlphaFold3 failed - {job_name}: {result.stderr}")
                return {'success': False, 'error_code': error_code, 'error_message': result.stderr}

            return {'success': True, 'output_dir': output_dir}

        except Exception as e:
            return {'success': False, 'error_code': 'EXECUTION_FAILED', 'error_message': str(e)}

    def clean_output_files(self, job_name, pipeline_id):
        output_dir = self._output_dir(pipeline_id, job_name)
        keep_files = [
            f"{job_name}_confidences.json",
            f"{job_name}_summary_confidences.json",
            f"{job_name}_model.cif"
        ]

        try:
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    if filename not in keep_files:
                        file_path = os.path.join(output_dir, filename)
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
        except Exception as e:
            self.logger.error(f"File cleanup error: {e}")

    def convert_cif_to_pdb(self, job_name, pipeline_id, iter, batch):
        try:
            output_dir = self._output_dir(pipeline_id, job_name)
            if not os.path.exists(output_dir):
                return {
                    'success': False,
                    'error_code': 'OUTPUT_DIR_NOT_FOUND',
                    'error_message': f"Output directory not found: {job_name}"
                }

            possible_paths = [
                os.path.join(output_dir, f"{job_name}_model.cif"),
                os.path.join(output_dir, f"{job_name.lower()}_model.cif"),
            ]

            input_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    input_path = path
                    break

            if not input_path:
                return {
                    'success': False,
                    'error_code': 'CIF_FILE_NOT_FOUND',
                    'error_message': f"CIF file not found: {job_name}"
                }

            filename = f"AF_{pipeline_id}_{iter}_{batch}.pdb"
            output_path = os.path.join(output_dir, filename)

            result = subprocess.run(
                [
                    "python3", "-c",
                    f"from pymol import cmd; cmd.quiet=1; cmd.load('{input_path}'); cmd.save('{output_path}'); cmd.delete('all')"
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return {
                    'success': False,
                    'error_code': 'CONVERSION_FAILED',
                    'error_message': f"PDB file creation failed: {output_path}"
                }

            return {'success': True, 'pdb_file_path': output_path, 'filename': filename}

        except subprocess.CalledProcessError as e:
            self.logger.error(f"CIF -> PDB conversion failed (subprocess): {e.stderr}")
            return {'success': False, 'error_code': 'CONVERSION_FAILED', 'error_message': e.stderr}
        except Exception as e:
            self.logger.error(f"CIF -> PDB conversion failed: {e}")
            return {'success': False, 'error_code': 'CONVERSION_FAILED', 'error_message': str(e)}

    def run(self, pipeline_id: int, sequence: str, iter: int, gpu_id: int, batch: int = 1, max_retries: int = 3):
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    cooldown = min(30 * (2 ** (attempt - 1)), 300)
                    self.logger.info(f"GPU cooldown: {cooldown}s (attempt {attempt + 1})")
                    time.sleep(cooldown)

                job_name = f"{pipeline_id}_{iter}_{batch}"

                input_result = self.create_alphafold_input(sequence, job_name, pipeline_id)
                if not input_result['success']:
                    return input_result

                af_result = self.run_alphafold(job_name, input_result['input_path'], gpu_id, pipeline_id)
                if not af_result['success']:
                    if self._is_retryable_error(af_result['error_code'], af_result['error_message']) and attempt < max_retries:
                        self.logger.warning(f"Retryable error, retrying: {af_result['error_code']}")
                        continue
                    return af_result

                self.clean_output_files(job_name, pipeline_id)

                conversion = self.convert_cif_to_pdb(job_name, pipeline_id, iter, batch)
                if not conversion['success']:
                    return conversion

                if attempt > 0:
                    self.logger.info(f"Retry succeeded - Pipeline: {pipeline_id}, attempt: {attempt + 1}")

                return {'success': True, 'pdb_file_path': conversion['pdb_file_path'], 'job_name': job_name}

            except Exception as e:
                if attempt < max_retries:
                    self.logger.error(f"Exception, retrying: {e}")
                    continue
                return {'success': False, 'error_code': 'MAX_RETRIES_EXCEEDED', 'error_message': str(e)}

        return {'success': False, 'error_code': 'MAX_RETRIES_EXCEEDED', 'error_message': f"Failed after {max_retries + 1} attempts"}

    def run_antibody(self, pipeline_id: int, fixed_chain_sequence: str, modified_chain_sequence: str,
                     cdr_chain_id: str, iter: int, gpu_id: int, batch: int = 1, max_retries: int = 3):
        """End-to-end 2-chain antibody fold: input -> AF3 -> CIF->PDB."""
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    cooldown = min(30 * (2 ** (attempt - 1)), 300)
                    self.logger.info(f"GPU cooldown: {cooldown}s (attempt {attempt + 1})")
                    time.sleep(cooldown)

                job_name = f"{pipeline_id}_{iter}_{batch}"

                input_result = self.create_alphafold_input_antibody(
                    fixed_chain_sequence, modified_chain_sequence, cdr_chain_id, job_name, pipeline_id
                )
                if not input_result['success']:
                    return input_result

                af_result = self.run_alphafold(job_name, input_result['input_path'], gpu_id, pipeline_id)
                if not af_result['success']:
                    if self._is_retryable_error(af_result['error_code'], af_result['error_message']) and attempt < max_retries:
                        self.logger.warning(f"Retryable error, retrying: {af_result['error_code']}")
                        continue
                    return af_result

                self.clean_output_files(job_name, pipeline_id)

                conversion = self.convert_cif_to_pdb(job_name, pipeline_id, iter, batch)
                if not conversion['success']:
                    return conversion

                if attempt > 0:
                    self.logger.info(f"Retry succeeded - Pipeline: {pipeline_id}, attempt: {attempt + 1}")

                return {'success': True, 'pdb_file_path': conversion['pdb_file_path'], 'job_name': job_name}

            except Exception as e:
                if attempt < max_retries:
                    self.logger.error(f"Exception, retrying: {e}")
                    continue
                return {'success': False, 'error_code': 'MAX_RETRIES_EXCEEDED', 'error_message': str(e)}

        return {'success': False, 'error_code': 'MAX_RETRIES_EXCEEDED', 'error_message': f"Failed after {max_retries + 1} attempts"}

    def _is_retryable_error(self, error_code: str, error_message: str) -> bool:
        non_retryable = ['INPUT_CREATION_FAILED', 'CIF_FILE_NOT_FOUND', 'CONVERSION_FAILED']
        return not any(pattern in error_code for pattern in non_retryable)