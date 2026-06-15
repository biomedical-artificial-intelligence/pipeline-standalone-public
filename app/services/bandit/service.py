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
    """AF → ZDOCK 파이프라인을 통과하는 단일 job 메타데이터"""
    job_id: int
    seq: str
    is_warmup: bool
    is_seed: bool = False
    select: Optional[int] = None
    aa: Optional[int] = None
    parent_seq: Optional[str] = None


class BanditStreamService:
    """
    UCB 스트리밍 학습 서비스.

    Warm-up phase:
    - seed_seq: update_from_sequence() - 전체 서열 기반 업데이트
    - 초기 3N: update_from_mutations() - seed 대비 mutation 위치만 업데이트
    - warm-up 전체 완료 후 best 서열 기준으로 n_zdock_workers * 2개 초기 search job 반환

    Search phase:
    - seq_meta로 (select, aa) 저장하여 out-of-order update 안전 처리
    - UCB 1-site mutation 탐색 (iter 10*N부터 UCB 전환)
    - top-K pool 기반 base 서열 선택 (80% best, 20% pool random)
    """

    def __init__(
        self,
        length: int,
        seed_seq: str,
        n_gpu_workers: int = 1,
        n_zdock_workers: int = 30,
        early_stop: bool = True,
        warmup_steps: int | None = None,
        es_window_size: int = 100,
        es_slope_horizon: int = 3,
        es_slope_eps: float = 0.01,
        es_patience_windows: int = 5,
        es_buffer_windows: int = 2,
        es_adaptive_buffer: bool = True,
        # [detector pass-through] 스케일 둔감 임계 모드. 기본값은 현행(absolute)이라
        # 동작 변화 없음. resume/실험에서 normalized/relative로 바꾸려면 값만 넘기면 됨.
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
        self.n_zdock_workers = n_zdock_workers
        # [early-stop warmup] 초기 랜덤 탐색 구간(=UCB random_phase=10*N) 제외.
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
        """seed 시퀀스 등록"""
        self.existing_seqs.add(self.seed_seq)
        self.warmup_total += 1
        return JobMeta(
            job_id=self._next_job_id(),
            seq=self.seed_seq,
            is_warmup=True,
            is_seed=True,
        )

    def register_initial_seqs(self, seqs: list) -> list[JobMeta]:
        """초기 3N 시퀀스 등록"""
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
        """히스토리 레코드에서 (select, aa) 역추론.
        mutated_position/mutated_aa 필드 우선, 없으면 parent_seq와 diff."""
        pos = rec.get("mutated_position")
        maa = rec.get("mutated_aa")
        if pos is not None and maa in aminoacid:
            return int(pos) - 1, aminoacid.index(maa)  # mutated_position은 1-based
        if parent_seq and len(parent_seq) == len(seq):
            for i, (p, c) in enumerate(zip(parent_seq, seq)):
                if p != c and c in aminoacid:
                    return i, aminoacid.index(c)
        return None

    def restore_from_history(self, history: list[dict]) -> int:
        """
        search-phase 히스토리(iter 오름차순, iter당 1서열)를 replay해 상태 복원.

        - warm-up(seed + 초기 3N)은 DB/파일에 미저장 → 생략. seed-aa 셀(~5%)만 미세
          영향(times=0으로 보여 resume 직후 몇 step 재탐색)이라 무시 가능.
        - detector(_stopper)에는 과거 점수를 주입하지 않는다 → fresh 윈도우로 시작해
          이어붙인 새 구간을 스스로 판단(즉시 재종료 방지).

        Returns: 복원된 iter_count.
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

            # 생성 시점에 t가 advance되던 것 모사(ucb bonus의 t 의존성 반영)
            self.M.t += 1

            sel_aa = self._infer_select_aa(rec, parent_seq, seq)
            if sel_aa is not None:
                self.M.select, self.M.aa = sel_aa
                self.M.update(norm)
            else:
                # parent 불명(예외) → 전체 서열 기반 업데이트로 폴백
                self.M.update_from_sequence(norm, string_to_index(seq))

            if score > self.best_score:
                self.best_score = score
                self.best_seq = seq

            self._add_to_top_k(score, seq)
            self.existing_seqs.add(seq)
            self.iter_count += 1

        # 이후 select_ac_one이 random_phase가 아닌 UCB 모드로 동작하도록 t 보정
        self.M.t = max(self.M.t, self.iter_count, self.M.random_phase_steps + 1)

        logger.info(
            f"[resume] restored {self.iter_count} iters | "
            f"best={self.best_score:.4f} seq={self.best_seq} | M.t={self.M.t}"
        )
        return self.iter_count

    def _add_to_top_k(self, score: float, seq: str):
        """top-K pool에 추가. K개 초과 시 최저 스코어 제거."""
        if len(self._top_k_pool) < self._TOP_K:
            heapq.heappush(self._top_k_pool, (score, seq))
        elif score > self._top_k_pool[0][0]:
            heapq.heapreplace(self._top_k_pool, (score, seq))

    def _pick_base_from_pool(self) -> str:
        """
        top-K pool에서 base 서열 선택.
        80% 확률로 best, 20% 확률로 pool 내 랜덤 선택 (exploration).
        """
        if not self._top_k_pool or random.random() < 0.8:
            return self.best_seq
        return random.choice(self._top_k_pool)[1]

    def on_result(self, job: JobMeta, score: float) -> list[JobMeta]:
        """
        ZDOCK 결과 수신 시 호출 (Updater에서 직렬로 호출).

        Returns:
            list[JobMeta]:
                - warm-up 진행 중: []
                - warm-up 완료 시: n_zdock_workers * 2개의 초기 search job
                - search phase: 다음 job 1개
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
                self.M.t = 1  # warmup t 리셋
                logger.info(
                    f"Warm-up complete ({self.warmup_total}) - "
                    f"best={self.best_score:.4f}, best_seq={self.best_seq}"
                )
                initial_jobs = [self._make_next_job(self.best_seq) for _ in range(self.n_zdock_workers * 2)]
                self.M.t = 1  # 초기 60개 생성 후 t 리셋 → iter 10*N부터 UCB 전환
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
            # BreakpointDetector가 내부에서 warmup_steps 구간을 제외한다.
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