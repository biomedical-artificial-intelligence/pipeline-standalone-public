import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class JobFileUtils:
    """파이프라인 파일 시스템 작업 유틸"""

    @staticmethod
    def save_warmup_sequences(
        output_dir: str,
        seed_seq: str,
        initial_seqs: list,
    ):
        try:
            warmup_path = os.path.join(output_dir, "warmup.json")
            data = {
                "seed_seq": seed_seq,
                "initial_seqs": initial_seqs,
                "total": 1 + len(initial_seqs),
            }
            with open(warmup_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Warmup sequences saved: {warmup_path} (total={data['total']})")
        except Exception as e:
            logger.error(f"Failed to save warmup sequences: {e}")

    @staticmethod
    def save_mutation_history(
        output_dir: str,
        iter: int,
        sequence: str,
        score: float,
        parent_seq: Optional[str] = None,
        parent_iter: Optional[int] = None,
    ):
        try:
            history_path = os.path.join(output_dir, "mutation_history.json")

            mutation_info = None
            if parent_seq and len(parent_seq) == len(sequence):
                for i, (p, c) in enumerate(zip(parent_seq, sequence)):
                    if p != c:
                        mutation_info = {
                            "mutated_position": i + 1,
                            "original_aa": p,
                            "mutated_aa": c,
                        }
                        break

            entry = {
                "iter": iter,
                "sequence": sequence,
                "score": score,
                "parent_seq": parent_seq,
                "parent_iter": parent_iter,
                **(mutation_info or {}),
            }

            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
            else:
                history = []

            history.append(entry)

            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save mutation history: {e}")
