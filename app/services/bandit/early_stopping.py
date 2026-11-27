"""
early_stopping.py

Early-stop decision logic, in four steps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Step A. Smoothed metrics
     - rolling_avg : mean of the last W scores
     - rolling_std : std of the last W scores  (drives the adaptive buffer)

 Step B. Plateau onset (both rules must hold to enter BUFFER)
     Rule 1 (slope):
         slope = rolling_avg[t] - rolling_avg[t - H]  (H = slope_horizon × W)
         triggers when slope < slope_eps
     Rule 2 (patience):
         triggers when best_rolling_avg has not improved for >= patience_windows × W

 Step C. Safety buffer
     Base length: buffer_windows × W
     adaptive=True: × 1.5 when the current rolling_std is in the top 25% of history
     → wait longer through volatile stretches

 Step D. Recovery check inside the buffer
     best_avg improves within the buffer (> slope_eps) → back to MONITORING
     buffer exhausted → STOPPED (final)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

State flow:

    MONITORING ──(Rule1 ∧ Rule2)──▶ BUFFER ──(buffer exhausted)──▶ STOPPED
         ▲                              │
         └──────(recovery in buffer)────┘

Parameters:
    window_size      (W)  : rolling avg/std window size            (default 100)
    slope_horizon    (SH) : slope comparison distance. H = SH × W  (default 3 → H=300)
    slope_eps        (ε)  : minimum meaningful improvement         (default 0.01)
    patience_windows (PW) : patience allowance. P = PW × W         (default 5 → P=500)
    buffer_windows   (BW) : buffer base length. buf = BW × W       (default 2 → buf=200)
    adaptive_buffer       : extend the buffer from rolling_std     (default True)

Minimum evals before anything can fire:
    slope computable : W + H = W(1 + SH)  →  400 by default
    patience fires   : P = PW × W          →  500 by default
    → earliest BUFFER entry : max(W+H, P) = 500
    → earliest STOPPED      : 500 + buf = 700
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
    """Early stop from plateau-onset detection plus a safety buffer."""

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

        # Minimum evals before slope is computable: W (fill window) + slope_H (build history)
        self.min_evals_slope = W + self.slope_H

        # ── Score tracking ─────────────────────────────────
        self._scores: deque[float] = deque(maxlen=W)

        # rolling avg history: the slope comparison needs at least slope_H+1 entries
        self._avg_history: deque[float] = deque(maxlen=self.slope_H + W + 1)

        # rolling std history: for the adaptive buffer percentile
        self._std_history: deque[float] = deque(maxlen=self.patience_P + self.buffer_base + 1)

        # ── Rule 2 (patience) ──────────────────────────────
        self._best_avg: float = -math.inf
        self._patience_counter: int = 0   # evals since best_avg last improved

        # ── State machine ──────────────────────────────────
        self._phase: str = _MONITORING
        self._n_evals: int = 0

        # ── Buffer state ───────────────────────────────────
        self._buffer_count: int = 0
        self._buffer_length: int = 0

        # ── Current metrics (for logs / stats) ────────────
        self.current_avg: float = math.nan
        self.current_std: float = math.nan
        self.slope: float = math.nan
        self._stop_reason: str = ""

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def update(self, score: float) -> None:
        """Take one score and update the state."""
        if self._phase == _STOPPED:
            return

        self._n_evals += 1
        self._scores.append(score)

        # Rolling metrics are undefined until the window is full
        if len(self._scores) < self.window_size:
            return

        # ── Step A: update rolling avg / std ────────────
        scores_list = list(self._scores)
        self.current_avg = sum(scores_list) / self.window_size
        self.current_std = math.sqrt(
            sum((s - self.current_avg) ** 2 for s in scores_list) / self.window_size
        )
        self._avg_history.append(self.current_avg)
        self._std_history.append(self.current_std)

        # ── Per-state update ────────────────────────────
        if self._phase == _MONITORING:
            self._update_monitoring()
        elif self._phase == _BUFFER:
            self._update_buffer()

    def should_stop(self) -> bool:
        """True when the search loop should stop."""
        return self._phase == _STOPPED

    def phase(self) -> str:
        """Current state: 'MONITORING' | 'BUFFER' | 'STOPPED'."""
        return self._phase

    def reason(self) -> str:
        """Reason string for the early stop."""
        return self._stop_reason

    def stats(self) -> dict:
        """Full current state as a dict."""
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
    # Internal methods
    # ──────────────────────────────────────────────────────

    def _rule1_slope(self) -> tuple[bool, float]:
        """
        Rule 1: slope test.
        Computable once avg_history holds more than slope_H entries.
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
        Rule 2: patience test.
        Tracks best_avg and returns True once it has not improved for patience_P evals.
        update_best=False leaves best_avg untouched (used to detect recovery in the buffer).
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
        Step C: buffer length.
        With adaptive=True, × 1.5 while the current rolling_std is in the top 25% (volatile).
        """
        base = self.buffer_base

        if not self.adaptive_buffer or len(self._std_history) < 4:
            return base

        stds_sorted = sorted(self._std_history)
        p75 = stds_sorted[int(len(stds_sorted) * 0.75)]
        multiplier = 1.5 if self.current_std >= p75 else 1.0
        length = int(base * multiplier)

        logger.debug(
            f"[EarlyStopper] buffer length: {length} eval "
            f"(std={self.current_std:.4f}, p75={p75:.4f}, ×{multiplier:.1f})"
        )
        return length

    def _update_monitoring(self) -> None:
        """
        MONITORING update.
        Rule1 ∧ Rule2 → enter BUFFER.
        """
        # Skip until there are enough evals for the slope test
        if self._n_evals < self.min_evals_slope:
            # Keep accumulating the patience counter
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
            # Step C: set the buffer length, then switch to BUFFER
            self._buffer_length = self._compute_buffer_length()
            self._buffer_count = 0
            self._phase = _BUFFER
            logger.info(
                f"[EarlyStopper] ▶ entering BUFFER (eval={self._n_evals}) | "
                f"slope={slope:+.4f} < eps={self.slope_eps} ✓ | "
                f"patience={self._patience_counter} ≥ {self.patience_P} ✓ | "
                f"buffer {self._buffer_length} eval (adaptive={self.adaptive_buffer})"
            )

    def _update_buffer(self) -> None:
        """
        BUFFER update.
        Step D: recovery → back to MONITORING, buffer exhausted → STOPPED.
        """
        self._buffer_count += 1

        # Step D: recovery check (did best_avg improve?)
        if self.current_avg > self._best_avg + self.slope_eps:
            self._best_avg = self.current_avg
            self._patience_counter = 0
            self._buffer_count = 0
            self._phase = _MONITORING
            logger.info(
                f"[EarlyStopper] back to MONITORING (eval={self._n_evals}) | "
                f"recovery: avg={self.current_avg:+.4f} > best_avg+eps "
                f"({self._best_avg + self.slope_eps:+.4f})"
            )
            return

        logger.debug(
            f"[EarlyStopper|BUFFER] {self._buffer_count}/{self._buffer_length} | "
            f"avg={self.current_avg:+.4f} | best_avg={self._best_avg:+.4f} | "
            f"std={self.current_std:.4f}"
        )

        # Buffer exhausted → STOPPED
        if self._buffer_count >= self._buffer_length:
            self._phase = _STOPPED
            self._stop_reason = (
                f"early stop confirmed (eval={self._n_evals}) | "
                f"buffer {self._buffer_length} eval exhausted, no recovery | "
                f"current_avg={self.current_avg:+.4f}, "
                f"best_avg={self._best_avg:+.4f}, "
                f"slope={self.slope:+.4f}"
            )
            logger.info(f"[EarlyStopper] ■ STOPPED — {self._stop_reason}")