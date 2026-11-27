import heapq
import random
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

from .utils import (
    init_ucb_param,
    normalize_score,
    string_to_index,
    indice_to_string,
    generate_random_sequence,
    aminoacid,
    aminoacid_len,
)
from .breakpoint_detector import BreakpointDetector

logger = logging.getLogger(__name__)

MAX_RESAMPLE = 100


@dataclass
class JobMeta:
    """Metadata for a single job passing through the fold → dock pipeline."""
    job_id: int
    seq: str
    is_warmup: bool
    is_seed: bool = False
    select: Optional[int] = None
    aa: Optional[int] = None
    parent_seq: Optional[str] = None


class BanditStreamService:
    """
    Streaming UCB learning service.

    Warm-up phase:
    - seed_seq: update_from_sequence() - update from the whole sequence
    - initial 3N: update_from_mutations() - update only positions mutated from the seed
    - once warm-up is complete, return n_dock_workers * 2 initial search jobs built
      from the best sequence

    Search phase:
    - (select, aa) is kept in seq_meta so out-of-order updates stay correct
    - UCB 1-site mutation search (UCB takes over from iter 10*N)
    - base sequence drawn from the top-K pool (80% best, 20% random from the pool)
    """

    def __init__(
        self,
        length: int,
        seed_seq: str,
        n_gpu_workers: int = 1,
        n_dock_workers: int = 30,
        early_stop: bool = True,
        warmup_steps: int | None = None,
        es_window_size: int = 100,
        es_slope_horizon: int = 3,
        es_slope_eps: float = 0.01,
        es_patience_windows: int = 5,
        es_buffer_windows: int = 2,
        es_adaptive_buffer: bool = True,
        # [detector pass-through] scale-insensitive threshold mode. The default is
        # absolute, so behaviour is unchanged; pass normalized/relative to switch.
        es_eps_mode: str = "absolute",
        es_relative_eps: float = 0.1,
        es_scale: float | None = None,
        es_center: float | None = None,
    ):
        self.M = init_ucb_param(length=length, iter=1)
        self.length = length
        self.seed_seq = seed_seq
        self.seed_idx = string_to_index(seed_seq)
        self.n_gpu_workers = n_gpu_workers
        self.n_dock_workers = n_dock_workers
        # [early-stop warmup] exclude the initial random-search phase (= UCB random_phase = 10*N).
        self.warmup_steps = warmup_steps if warmup_steps is not None else 10 * length

        self.existing_seqs: set = set()
        self.seq_meta: dict = {}

        self._job_counter = 0
        self.best_score = -1e18
        self.best_seq = None
        self.iter_count = 0

        self.warmup_total = 0
        self.warmup_done = 0
        self.warmup_complete = False

        self._top_k_pool: list = []  # min-heap of (score, seq)
        self._TOP_K = 10

        self._stopper: BreakpointDetector | None = (
            BreakpointDetector(
                window_size=es_window_size,
                slope_horizon=es_slope_horizon,
                slope_eps=es_slope_eps,
                patience_windows=es_patience_windows,
                buffer_windows=es_buffer_windows,
                adaptive_buffer=es_adaptive_buffer,
                warmup_steps=self.warmup_steps,
                eps_mode=es_eps_mode,
                relative_eps=es_relative_eps,
                scale=es_scale,
                center=es_center,
            )
            if early_stop
            else None
        )

    @property
    def is_early_stopped(self) -> bool:
        return self._stopper is not None and self._stopper.should_stop()

    def register_seed(self) -> JobMeta:
        """Register the seed sequence."""
        self.existing_seqs.add(self.seed_seq)
        self.warmup_total += 1
        return JobMeta(
            job_id=self._next_job_id(),
            seq=self.seed_seq,
            is_warmup=True,
            is_seed=True,
        )

    def register_initial_seqs(self, seqs: list) -> list[JobMeta]:
        """Register the initial 3N sequences."""
        jobs = []
        for seq in seqs:
            self.existing_seqs.add(seq)
            self.warmup_total += 1
            jobs.append(JobMeta(
                job_id=self._next_job_id(),
                seq=seq,
                is_warmup=True,
                is_seed=False,
            ))
        return jobs

    @staticmethod
    def _infer_select_aa(rec: dict, parent_seq: Optional[str], seq: str):
        """Infer (select, aa) from a history record.
        Prefers the mutated_position/mutated_aa fields, else diffs against parent_seq."""
        pos = rec.get("mutated_position")
        maa = rec.get("mutated_aa")
        if pos is not None and maa in aminoacid:
            return int(pos) - 1, aminoacid.index(maa)  # mutated_position is 1-based
        if parent_seq and len(parent_seq) == len(seq):
            for i, (p, c) in enumerate(zip(parent_seq, seq)):
                if p != c and c in aminoacid:
                    return i, aminoacid.index(c)
        return None

    def restore_from_history(self, history: list[dict]) -> int:
        """
        Restore state by replaying the search-phase history (ascending iter, one
        sequence per iter).

        - warm-up (seed + initial 3N) is not persisted to the DB or to files, so it is
          skipped. Only the seed-aa cells (~5%) are affected: they read as untried
          (times=0), costing a few re-explored steps right after resume.
        - past scores are not fed to the detector (_stopper); it starts on a fresh
          window and judges the appended stretch on its own (no immediate re-stop).

        Returns: the restored iter_count.
        """
        self.warmup_complete = True
        self.warmup_done = self.warmup_total

        for rec in history:
            seq = rec.get("sequence")
            score = rec.get("score")
            if not seq or score is None:
                continue

            parent_seq = rec.get("parent_seq")
            norm = normalize_score(score)

            # Mimic the t advance that happened at generation time (ucb bonus depends on t)
            self.M.t += 1

            sel_aa = self._infer_select_aa(rec, parent_seq, seq)
            if sel_aa is not None:
                self.M.select, self.M.aa = sel_aa
                self.M.update(norm)
            else:
                # Unknown parent → fall back to a whole-sequence update
                self.M.update_from_sequence(norm, string_to_index(seq))

            if score > self.best_score:
                self.best_score = score
                self.best_seq = seq

            self._add_to_top_k(score, seq)
            self.existing_seqs.add(seq)
            self.iter_count += 1

        # Bump t so select_ac_one runs in UCB mode rather than the random phase
        self.M.t = max(self.M.t, self.iter_count, self.M.random_phase_steps + 1)

        logger.info(
            f"[resume] restored {self.iter_count} iters | "
            f"best={self.best_score:.4f} seq={self.best_seq} | M.t={self.M.t}"
        )
        return self.iter_count

    def _add_to_top_k(self, score: float, seq: str):
        """Add to the top-K pool, evicting the lowest score once it exceeds K."""
        if len(self._top_k_pool) < self._TOP_K:
            heapq.heappush(self._top_k_pool, (score, seq))
        elif score > self._top_k_pool[0][0]:
            heapq.heapreplace(self._top_k_pool, (score, seq))

    def _pick_base_from_pool(self) -> str:
        """
        Pick the base sequence from the top-K pool:
        best with probability 80%, a random pool entry with 20% (exploration).
        """
        if not self._top_k_pool or random.random() < 0.8:
            return self.best_seq
        return random.choice(self._top_k_pool)[1]

    def on_result(self, job: JobMeta, score: float) -> list[JobMeta]:
        """
        Called when a docking result arrives (serially, from the Updater).

        Returns:
            list[JobMeta]:
                - warm-up in progress: []
                - warm-up just completed: n_dock_workers * 2 initial search jobs
                - search phase: the next single job
        """
        norm_score = normalize_score(score)

        if score > self.best_score:
            self.best_score = score
            self.best_seq = job.seq
            logger.info(f"New best - iter {self.iter_count}: {self.best_seq} score={self.best_score:.4f}")

        self._add_to_top_k(score, job.seq)

        # ----------------------------------------------------------------
        # Warm-up phase
        # ----------------------------------------------------------------
        if job.is_warmup:
            seq_idx = string_to_index(job.seq)

            if job.is_seed:
                self.M.update_from_sequence(norm_score, seq_idx)
                logger.info(f"Warm-up seed done - score={score:.4f}")
            else:
                self.M.update_from_mutations(norm_score, self.seed_idx, seq_idx)

            self.warmup_done += 1

            if self.warmup_done == self.warmup_total:
                self.warmup_complete = True
                self.M.t = 1  # reset t after warm-up
                logger.info(
                    f"Warm-up complete ({self.warmup_total}) - "
                    f"best={self.best_score:.4f}, best_seq={self.best_seq}"
                )
                initial_jobs = [self._make_next_job(self.best_seq) for _ in range(self.n_dock_workers * 2)]
                self.M.t = 1  # reset t after the initial batch → UCB takes over at iter 10*N
                return initial_jobs

            return []

        # ----------------------------------------------------------------
        # Search phase
        # ----------------------------------------------------------------
        self.iter_count += 1
        sel, aa = self.seq_meta.get(job.seq, (job.select, job.aa))
        self.M.select = sel
        self.M.aa = aa
        self.M.update(norm_score)

        if self._stopper is not None:
            # BreakpointDetector excludes the warmup_steps range internally.
            self._stopper.update(score)

        logger.info(
            f"Iter {self.iter_count} | score={score:.4f} | best={self.best_score:.4f} | "
            f"t={self.M.t} | seq={job.seq}"
            + (f" | ES[{self._stopper.phase()} buf={self._stopper.stats()['buffer_count']}/{self._stopper.stats()['buffer_length']}]"
               if self._stopper is not None and self._stopper.phase() == 'BUFFER'
               else "")
        )

        base_seq = self._pick_base_from_pool()
        return [self._make_next_job(base_seq)]

    def _make_next_job(self, base_seq: str) -> JobMeta:
        base_idx = string_to_index(base_seq)
        new_seq, select, aa = self._generate_next(base_idx)

        self.seq_meta[new_seq] = (select, aa)

        return JobMeta(
            job_id=self._next_job_id(),
            seq=new_seq,
            is_warmup=False,
            select=select,
            aa=aa,
            parent_seq=base_seq,
        )

    def _generate_next(self, base_idx: list) -> tuple[str, int, int]:
        for attempt in range(MAX_RESAMPLE):
            action = self.M.select_ac_one(
                base_idx.copy(),
                advance_time=(attempt == 0),
            )
            candidate = indice_to_string(action)
            if candidate not in self.existing_seqs:
                self.existing_seqs.add(candidate)
                return candidate, int(self.M.select), int(self.M.aa)

        logger.warning("Resample limit reached, applying random fallback")
        idx = base_idx[:]
        pos = random.randrange(self.length)
        old = idx[pos]
        aa = random.randrange(aminoacid_len)
        if aa == old:
            aa = (aa + 1) % aminoacid_len
        idx[pos] = aa
        candidate = indice_to_string(idx)
        self.existing_seqs.add(candidate)
        return candidate, pos, aa

    def _next_job_id(self) -> int:
        jid = self._job_counter
        self._job_counter += 1
        return jid