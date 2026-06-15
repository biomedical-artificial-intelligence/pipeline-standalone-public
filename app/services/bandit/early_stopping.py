"""
early_stopping.py

4단계 실무 절차 기반 조기 종료 판단기.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Step A. 스무딩 지표
     - rolling_avg : 최근 W개 평균
     - rolling_std : 최근 W개 표준편차  (adaptive buffer 조정용)

 Step B. 정체 시작점 판정 (두 규칙 모두 충족 시 BUFFER 진입)
     Rule 1 (기울기):
         slope = rolling_avg[t] - rolling_avg[t - H]  (H = slope_horizon × W)
         slope < slope_eps 이면 트리거
     Rule 2 (patience):
         best_rolling_avg 미갱신 구간 >= patience_windows × W 이면 트리거

 Step C. 안전 버퍼
     기본 길이: buffer_windows × W
     adaptive=True: 현재 rolling_std 가 이력의 상위 25%이면 × 1.5 연장
     → 변동성이 높은 구간에선 더 기다린다

 Step D. 버퍼 구간 재상승 확인
     버퍼 내 best_avg 갱신 (> slope_eps) → MONITORING 복귀 (정체 해제)
     버퍼 소진 → STOPPED (최종 종료)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

상태 흐름:

    MONITORING ──(Rule1 ∧ Rule2)──▶ BUFFER ──(버퍼 소진)──▶ STOPPED
         ▲                              │
         └──────(버퍼 중 재상승)────────┘

파라미터:
    window_size      (W)  : rolling avg/std 윈도우 크기          (default 100)
    slope_horizon    (SH) : 기울기 비교 거리 배수. H = SH × W    (default 3 → H=300)
    slope_eps        (ε)  : 의미 있는 개선의 최소값              (default 0.01)
    patience_windows (PW) : patience 허용 배수. P = PW × W       (default 5 → P=500)
    buffer_windows   (BW) : 버퍼 기본 길이 배수. buf = BW × W    (default 2 → buf=200)
    adaptive_buffer       : rolling_std 기반 버퍼 동적 연장      (default True)

최소 발동 eval:
    slope 판단 가능: W + H = W(1 + SH)  →  기본 400
    patience 발동 : P = PW × W           →  기본 500
    → 첫 BUFFER 진입 가능 시점: max(W+H, P) = 500
    → 첫 STOPPED 가능 시점   : 500 + buf = 700
"""

from __future__ import annotations

import logging
import math
from collections import deque

logger = logging.getLogger(__name__)

_MONITORING = "MONITORING"
_BUFFER = "BUFFER"
_STOPPED = "STOPPED"


class EarlyStopper:
    """정체 시작점 탐지 + 안전 버퍼 기반 조기 종료 판단기."""

    def __init__(
        self,
        window_size: int = 100,
        slope_horizon: int = 3,
        slope_eps: float = 0.01,
        patience_windows: int = 5,
        buffer_windows: int = 2,
        adaptive_buffer: bool = True,
    ) -> None:
        W = window_size
        self.window_size = W
        self.slope_H = slope_horizon * W
        self.slope_eps = slope_eps
        self.patience_P = patience_windows * W
        self.buffer_base = buffer_windows * W
        self.adaptive_buffer = adaptive_buffer

        # slope 판단 가능 최소 eval: W(윈도우 채움) + slope_H(이력 형성)
        self.min_evals_slope = W + self.slope_H

        # ── 점수 추적 ──────────────────────────────────────
        self._scores: deque[float] = deque(maxlen=W)

        # rolling avg 이력: slope 비교를 위해 최소 slope_H+1개 필요
        self._avg_history: deque[float] = deque(maxlen=self.slope_H + W + 1)

        # rolling std 이력: adaptive buffer percentile 계산용
        self._std_history: deque[float] = deque(maxlen=self.patience_P + self.buffer_base + 1)

        # ── Rule 2 (patience) ──────────────────────────────
        self._best_avg: float = -math.inf
        self._patience_counter: int = 0   # best_avg 미갱신 eval 누적 수

        # ── 상태 기계 ──────────────────────────────────────
        self._phase: str = _MONITORING
        self._n_evals: int = 0

        # ── 버퍼 상태 ──────────────────────────────────────
        self._buffer_count: int = 0
        self._buffer_length: int = 0

        # ── 현재 지표 (로그 / stats 용) ───────────────────
        self.current_avg: float = math.nan
        self.current_std: float = math.nan
        self.slope: float = math.nan
        self._stop_reason: str = ""

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def update(self, score: float) -> None:
        """score 1개를 받아 상태를 갱신한다."""
        if self._phase == _STOPPED:
            return

        self._n_evals += 1
        self._scores.append(score)

        # 윈도우가 가득 차기 전엔 rolling 지표 계산 불가
        if len(self._scores) < self.window_size:
            return

        # ── Step A: rolling avg / std 갱신 ──────────────
        scores_list = list(self._scores)
        self.current_avg = sum(scores_list) / self.window_size
        self.current_std = math.sqrt(
            sum((s - self.current_avg) ** 2 for s in scores_list) / self.window_size
        )
        self._avg_history.append(self.current_avg)
        self._std_history.append(self.current_std)

        # ── 상태별 갱신 ─────────────────────────────────
        if self._phase == _MONITORING:
            self._update_monitoring()
        elif self._phase == _BUFFER:
            self._update_buffer()

    def should_stop(self) -> bool:
        """True이면 search loop를 종료해야 한다."""
        return self._phase == _STOPPED

    def phase(self) -> str:
        """현재 상태 반환: 'MONITORING' | 'BUFFER' | 'STOPPED'."""
        return self._phase

    def reason(self) -> str:
        """조기 종료 사유 문자열 반환."""
        return self._stop_reason

    def stats(self) -> dict:
        """현재 전체 상태 딕셔너리 반환."""
        return {
            "n_evals":          self._n_evals,
            "phase":            self._phase,
            "current_avg":      self.current_avg,
            "current_std":      self.current_std,
            "slope":            self.slope,
            "best_avg":         self._best_avg,
            "patience_counter": self._patience_counter,
            "patience_limit":   self.patience_P,
            "buffer_count":     self._buffer_count,
            "buffer_length":    self._buffer_length,
            "stopped_early":    self._phase == _STOPPED,
            "stop_reason":      self._stop_reason,
        }

    # ──────────────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────────────

    def _rule1_slope(self) -> tuple[bool, float]:
        """
        Rule 1: 기울기 판정.
        avg_history가 slope_H+1개 이상 쌓여야 계산 가능.
        returns (triggered, slope)
        """
        if len(self._avg_history) <= self.slope_H:
            return False, math.nan

        slope = self._avg_history[-1] - self._avg_history[-(self.slope_H + 1)]
        self.slope = slope
        triggered = slope < self.slope_eps
        return triggered, slope

    def _rule2_patience(self, *, update_best: bool = True) -> bool:
        """
        Rule 2: patience 판정.
        best_avg를 추적하며 미갱신 구간이 patience_P 이상이면 True.
        update_best=False이면 best_avg를 갱신하지 않는다 (버퍼 구간 재상승 감지용).
        """
        if self.current_avg > self._best_avg + self.slope_eps:
            if update_best:
                self._best_avg = self.current_avg
                self._patience_counter = 0
            return False

        self._patience_counter += 1
        return self._patience_counter >= self.patience_P

    def _compute_buffer_length(self) -> int:
        """
        Step C: 버퍼 길이 계산.
        adaptive=True이면 현재 rolling_std가 이력의 상위 25%(고변동) 구간일 때 × 1.5.
        """
        base = self.buffer_base

        if not self.adaptive_buffer or len(self._std_history) < 4:
            return base

        stds_sorted = sorted(self._std_history)
        p75 = stds_sorted[int(len(stds_sorted) * 0.75)]
        multiplier = 1.5 if self.current_std >= p75 else 1.0
        length = int(base * multiplier)

        logger.debug(
            f"[EarlyStopper] 버퍼 길이 결정: {length} eval "
            f"(std={self.current_std:.4f}, p75={p75:.4f}, ×{multiplier:.1f})"
        )
        return length

    def _update_monitoring(self) -> None:
        """
        MONITORING 상태 갱신.
        Rule1 ∧ Rule2 충족 시 → BUFFER 진입.
        """
        # slope 판단 가능 최소 eval 미달이면 스킵
        if self._n_evals < self.min_evals_slope:
            # patience 카운터는 계속 누적
            self._rule2_patience()
            return

        r1_triggered, slope = self._rule1_slope()
        r2_triggered = self._rule2_patience()

        logger.debug(
            f"[EarlyStopper|MONITOR] eval={self._n_evals} | "
            f"avg={self.current_avg:+.4f} | std={self.current_std:.4f} | "
            f"slope={slope:+.4f} R1={'✓' if r1_triggered else '✗'} | "
            f"patience={self._patience_counter}/{self.patience_P} R2={'✓' if r2_triggered else '✗'}"
        )

        if r1_triggered and r2_triggered:
            # Step C: 버퍼 길이 결정 후 BUFFER 전환
            self._buffer_length = self._compute_buffer_length()
            self._buffer_count = 0
            self._phase = _BUFFER
            logger.info(
                f"[EarlyStopper] ▶ BUFFER 진입 (eval={self._n_evals}) | "
                f"slope={slope:+.4f} < eps={self.slope_eps} ✓ | "
                f"patience={self._patience_counter} ≥ {self.patience_P} ✓ | "
                f"버퍼 {self._buffer_length} eval (adaptive={self.adaptive_buffer})"
            )

    def _update_buffer(self) -> None:
        """
        BUFFER 상태 갱신.
        Step D: 재상승 감지 시 MONITORING 복귀, 버퍼 소진 시 STOPPED.
        """
        self._buffer_count += 1

        # Step D: 재상승 확인 (best_avg 갱신 여부)
        if self.current_avg > self._best_avg + self.slope_eps:
            self._best_avg = self.current_avg
            self._patience_counter = 0
            self._buffer_count = 0
            self._phase = _MONITORING
            logger.info(
                f"[EarlyStopper] MONITORING 복귀 (eval={self._n_evals}) | "
                f"재상승 감지: avg={self.current_avg:+.4f} > best_avg+eps "
                f"({self._best_avg + self.slope_eps:+.4f})"
            )
            return

        logger.debug(
            f"[EarlyStopper|BUFFER] {self._buffer_count}/{self._buffer_length} | "
            f"avg={self.current_avg:+.4f} | best_avg={self._best_avg:+.4f} | "
            f"std={self.current_std:.4f}"
        )

        # 버퍼 소진 → STOPPED
        if self._buffer_count >= self._buffer_length:
            self._phase = _STOPPED
            self._stop_reason = (
                f"조기 종료 확정 (eval={self._n_evals}) | "
                f"버퍼 {self._buffer_length} eval 소진, 재상승 없음 | "
                f"current_avg={self.current_avg:+.4f}, "
                f"best_avg={self._best_avg:+.4f}, "
                f"slope={self.slope:+.4f}"
            )
            logger.info(f"[EarlyStopper] ■ STOPPED — {self._stop_reason}")