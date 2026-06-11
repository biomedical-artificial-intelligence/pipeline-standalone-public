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

        self.min_evals_slope = W + self.slope_H

        self._scores: deque[float] = deque(maxlen=W)
        self._avg_history: deque[float] = deque(maxlen=self.slope_H + W + 1)
        self._std_history: deque[float] = deque(maxlen=self.patience_P + self.buffer_base + 1)

        self._best_avg: float = -math.inf
        self._patience_counter: int = 0

        self._phase: str = _MONITORING
        self._n_evals: int = 0

        self._buffer_count: int = 0
        self._buffer_length: int = 0

        self.current_avg: float = math.nan
        self.current_std: float = math.nan
        self.slope: float = math.nan
        self._stop_reason: str = ""

    def update(self, score: float) -> None:
        if self._phase == _STOPPED:
            return

        self._n_evals += 1
        self._scores.append(score)

        if len(self._scores) < self.window_size:
            return

        scores_list = list(self._scores)
        self.current_avg = sum(scores_list) / self.window_size
        self.current_std = math.sqrt(
            sum((s - self.current_avg) ** 2 for s in scores_list) / self.window_size
        )
        self._avg_history.append(self.current_avg)
        self._std_history.append(self.current_std)

        if self._phase == _MONITORING:
            self._update_monitoring()
        elif self._phase == _BUFFER:
            self._update_buffer()

    def should_stop(self) -> bool:
        return self._phase == _STOPPED

    def phase(self) -> str:
        return self._phase

    def reason(self) -> str:
        return self._stop_reason

    def stats(self) -> dict:
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

    def _rule1_slope(self) -> tuple[bool, float]:
        if len(self._avg_history) <= self.slope_H:
            return False, math.nan

        slope = self._avg_history[-1] - self._avg_history[-(self.slope_H + 1)]
        self.slope = slope
        triggered = slope < self.slope_eps
        return triggered, slope

    def _rule2_patience(self, *, update_best: bool = True) -> bool:
        if self.current_avg > self._best_avg + self.slope_eps:
            if update_best:
                self._best_avg = self.current_avg
                self._patience_counter = 0
            return False

        self._patience_counter += 1
        return self._patience_counter >= self.patience_P

    def _compute_buffer_length(self) -> int:
        base = self.buffer_base

        if not self.adaptive_buffer or len(self._std_history) < 4:
            return base

        stds_sorted = sorted(self._std_history)
        p75 = stds_sorted[int(len(stds_sorted) * 0.75)]
        multiplier = 1.5 if self.current_std >= p75 else 1.0
        length = int(base * multiplier)

        return length

    def _update_monitoring(self) -> None:
        if self._n_evals < self.min_evals_slope:
            self._rule2_patience()
            return

        r1_triggered, slope = self._rule1_slope()
        r2_triggered = self._rule2_patience()

        if r1_triggered and r2_triggered:
            self._buffer_length = self._compute_buffer_length()
            self._buffer_count = 0
            self._phase = _BUFFER
            logger.info(
                f"Early stopper BUFFER entered (eval={self._n_evals}) | "
                f"slope={slope:+.4f} < eps={self.slope_eps} | "
                f"patience={self._patience_counter} >= {self.patience_P} | "
                f"buffer={self._buffer_length} eval"
            )

    def _update_buffer(self) -> None:
        self._buffer_count += 1

        if self.current_avg > self._best_avg + self.slope_eps:
            self._best_avg = self.current_avg
            self._patience_counter = 0
            self._buffer_count = 0
            self._phase = _MONITORING
            logger.info(
                f"Early stopper MONITORING resumed (eval={self._n_evals}) | "
                f"avg={self.current_avg:+.4f}"
            )
            return

        if self._buffer_count >= self._buffer_length:
            self._phase = _STOPPED
            self._stop_reason = (
                f"early stop confirmed (eval={self._n_evals}) | "
                f"buffer={self._buffer_length} exhausted | "
                f"current_avg={self.current_avg:+.4f}, best_avg={self._best_avg:+.4f}, slope={self.slope:+.4f}"
            )
            logger.info(f"Early stopper STOPPED - {self._stop_reason}")