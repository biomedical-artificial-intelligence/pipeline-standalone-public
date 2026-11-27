"""
breakpoint_detector.py

Breakpoint (early-stop) detection for the UCB docking search loop.

Layers:
  - core decision logic : early_stopping.EarlyStopper (used unmodified)
  - added here          : (1) warm-up exclusion for the initial random-search phase
                          (2) scale-insensitive thresholds (absolute / relative / normalized)

Standard library only (no numpy).

────────────────────────────────────────────────────────────────────
Threshold (slope_eps) scale problem
  EarlyStopper's slope_eps is a threshold on the absolute slope of the moving
  average. Score scales differ between datasets (synthetic amplitude ~10 vs a
  measured docking mean ~1.0 whose moving average moves by ~0.05), so one
  slope_eps that fits the first is far too large for the second, and a plateau
  is detected where there is none → early stop fires.

  eps_mode handles this:
    - "absolute"   : use slope_eps as given (legacy behaviour, default)
    - "relative"   : derive it from the data scale
                     eff_eps = relative_eps × (scale / sqrt(window_size))
                     scale = score standard deviation (warm-up or explicit)
    - "normalized" : feed (x - center) / scale to the detector, which makes
                     slope_eps scale-invariant (normalized std ≈ 1)

  scale/center are calibrated from warm-up samples (online) or passed in
  explicitly (offline: whole-array statistics recommended).
────────────────────────────────────────────────────────────────────

Conventions
  - Higher scores are better (docking score, matching best_avg tracking).
  - warm-up: the first N UCB steps are trendless random search, so they are
    excluded from detection.

Online use (small-scale real data: normalized recommended):
    from breakpoint_detector import BreakpointDetector
    det = BreakpointDetector(window_size=100, warmup_steps=500,
                             eps_mode="normalized")     # auto-calibrated from warm-up
    for step in range(max_steps):
        score = run_ucb_dock_step(...)
        if det.update(score):
            print("early stop:", det.reason()); break

Offline use:
    from breakpoint_detector import find_breakpoint
    out = find_breakpoint(scores, warmup_steps=500, eps_mode="relative", relative_eps=0.1)
    print(out["stop_step"], out["effective_slope_eps"])
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
from statistics import fmean, pstdev
from typing import Iterable, Optional

from .early_stopping import EarlyStopper

logger = logging.getLogger(__name__)

ABSOLUTE = "absolute"
RELATIVE = "relative"
NORMALIZED = "normalized"
_MODES = (ABSOLUTE, RELATIVE, NORMALIZED)


class BreakpointDetector:
    """
    EarlyStopper plus warm-up exclusion and scale-insensitive thresholds.

    Parameters
        window_size      (W)  : rolling avg/std window size             (default 100)
        slope_horizon    (SH) : slope comparison distance. H = SH × W    (default 3)
        slope_eps        (ε)  : minimum meaningful improvement           (default 0.01)
                                · eps_mode="absolute"  → used as given
                                · eps_mode="normalized"→ ε on the normalized (std≈1) scale
        patience_windows (PW) : patience allowance. P = PW × W           (default 5)
        buffer_windows   (BW) : safety buffer base length. buf = BW × W  (default 2)
        adaptive_buffer       : extend the buffer from rolling_std       (default True)
        warmup_steps     (N)  : length of the excluded random-search run (default 500)

      Scale-insensitive options:
        eps_mode              : "absolute" | "relative" | "normalized"  (default "absolute")
        relative_eps          : factor used when eps_mode="relative"     (default 0.1)
        scale                 : data scale (std). None → calibrate from warm-up
        center                : data center (mean). None → calibrate from warm-up
        min_calibration       : minimum samples needed to calibrate      (default 30)
    """

    def __init__(
        self,
        window_size: int = 100,
        slope_horizon: int = 3,
        slope_eps: float = 0.01,
        patience_windows: int = 5,
        buffer_windows: int = 2,
        adaptive_buffer: bool = True,
        warmup_steps: int = 500,
        *,
        eps_mode: str = ABSOLUTE,
        relative_eps: float = 0.1,
        scale: Optional[float] = None,
        center: Optional[float] = None,
        min_calibration: int = 30,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if eps_mode not in _MODES:
            raise ValueError(f"eps_mode must be one of {_MODES}")

        self.warmup_steps = warmup_steps
        self.eps_mode = eps_mode
        self.relative_eps = relative_eps
        self.min_calibration = min_calibration

        # EarlyStopper kwargs (slope_eps is finalized after calibration)
        self._es_kwargs = dict(
            window_size=window_size,
            slope_horizon=slope_horizon,
            slope_eps=slope_eps,
            patience_windows=patience_windows,
            buffer_windows=buffer_windows,
            adaptive_buffer=adaptive_buffer,
        )
        self._base_slope_eps = slope_eps
        self._window_size = window_size

        self._user_scale = scale
        self._user_center = center
        self._scale = 1.0           # data scale used by relative/normalized modes
        self._center = 0.0
        self.effective_slope_eps = slope_eps

        self._warm: list[float] = []   # raw warm-up samples for calibration
        self._step = 0
        self._stop_step: Optional[int] = None

        # absolute needs no calibration, so build the detector right away.
        # relative/normalized build it after warm-up calibration.
        self._stopper: Optional[EarlyStopper] = None
        if eps_mode == ABSOLUTE and scale is None and center is None:
            self._stopper = EarlyStopper(**self._es_kwargs)
        # With explicit scale/center, absolute still applies the transform (rarely needed)

    # ──────────────────────────────────────────────────────────────
    # Internal: scale calibration + detector construction
    # ──────────────────────────────────────────────────────────────
    def _calibrate_and_build(self) -> None:
        """Called once at the end of warm-up (or immediately): fix scale/center, build EarlyStopper."""
        # Explicit scale/center wins, otherwise estimate from warm-up samples
        if self._user_scale is not None:
            scale = float(self._user_scale)
            center = float(self._user_center) if self._user_center is not None else 0.0
        elif len(self._warm) >= self.min_calibration:
            center = fmean(self._warm)
            scale = pstdev(self._warm)
        else:
            # Too few calibration samples → fall back to absolute
            logger.warning(
                "[BreakpointDetector] too few calibration samples (%d < %d) → eps_mode=absolute",
                len(self._warm), self.min_calibration,
            )
            self.eps_mode = ABSOLUTE
            center, scale = 0.0, 1.0

        if not scale or scale <= 0 or math.isnan(scale):
            scale = 1.0   # guard against zero variance

        self._center, self._scale = center, scale
        kw = dict(self._es_kwargs)

        if self.eps_mode == RELATIVE:
            # Natural unit of the moving-average slope ≈ scale / sqrt(W)
            kw["slope_eps"] = self.relative_eps * scale / math.sqrt(self._window_size)
        elif self.eps_mode == NORMALIZED:
            # On the normalized (std≈1) scale, base slope_eps applies as is
            kw["slope_eps"] = self._base_slope_eps
        # absolute: base as is

        self.effective_slope_eps = kw["slope_eps"]
        self._stopper = EarlyStopper(**kw)

    def _transform(self, score: float) -> float:
        """Apply (x - center)/scale in normalized mode only, else return the raw value."""
        if self.eps_mode == NORMALIZED:
            return (score - self._center) / self._scale
        return score

    # ──────────────────────────────────────────────────────────────
    # Online API
    # ──────────────────────────────────────────────────────────────
    def update(self, score: float) -> bool:
        """
        Feed one score and return whether the search should stop now.
        Warm-up scores (the first N steps) are not used for detection and always
        return False, but relative/normalized modes calibrate the scale from them.
        """
        i = self._step
        self._step += 1
        score = float(score)

        if i < self.warmup_steps:
            if self.eps_mode != ABSOLUTE and self._user_scale is None:
                self._warm.append(score)     # collected for calibration
            return False

        # On the first detection step, calibrate and build if not built yet
        if self._stopper is None:
            self._calibrate_and_build()

        self._stopper.update(self._transform(score))
        stop = self._stopper.should_stop()
        if stop and self._stop_step is None:
            self._stop_step = i
        return stop

    def should_stop(self) -> bool:
        return self._stopper.should_stop() if self._stopper is not None else False

    def phase(self) -> str:
        return self._stopper.phase() if self._stopper is not None else "MONITORING"

    def reason(self) -> str:
        return self._stopper.reason() if self._stopper is not None else ""

    @property
    def stop_step(self) -> Optional[int]:
        """Absolute step index where the stop was confirmed (None if not stopped)."""
        return self._stop_step

    @property
    def n_steps(self) -> int:
        return self._step

    def stats(self) -> dict:
        """EarlyStopper internals plus warm-up / scale / threshold info."""
        s = dict(self._stopper.stats()) if self._stopper is not None else {}
        s.update(
            warmup_steps=self.warmup_steps,
            total_steps=self._step,
            stop_step=self._stop_step,
            eps_mode=self.eps_mode,
            scale=self._scale,
            center=self._center,
            effective_slope_eps=self.effective_slope_eps,
        )
        return s


def find_breakpoint(
    scores: Iterable[float],
    *,
    warmup_steps: int = 500,
    eps_mode: str = ABSOLUTE,
    relative_eps: float = 0.1,
    calibrate: str = "auto",
    window_size: int = 100,
    slope_horizon: int = 3,
    slope_eps: float = 0.01,
    patience_windows: int = 5,
    buffer_windows: int = 2,
    adaptive_buffer: bool = True,
    scale: Optional[float] = None,
    center: Optional[float] = None,
) -> dict:
    """
    Offline helper: find the breakpoint of a score sequence in one pass.
    Iteration stops as soon as the stop is confirmed.

    calibrate:
        "auto"/"global" : in relative/normalized modes, compute scale/center from
                          whole-array statistics (robust, the offline default).
                          Explicit scale=/center= take precedence.
        "warmup"        : calibrate from warm-up samples only (same as online).

    Returns dict: stop_step, stopped, phase, reason, effective_slope_eps, scale, center, stats
    """
    seq = [float(x) for x in scores]

    if eps_mode in (RELATIVE, NORMALIZED) and scale is None and calibrate in ("auto", "global"):
        center = fmean(seq) if center is None else center
        sc = pstdev(seq)
        scale = sc if sc and sc > 0 else 1.0

    det = BreakpointDetector(
        window_size=window_size,
        slope_horizon=slope_horizon,
        slope_eps=slope_eps,
        patience_windows=patience_windows,
        buffer_windows=buffer_windows,
        adaptive_buffer=adaptive_buffer,
        warmup_steps=warmup_steps,
        eps_mode=eps_mode,
        relative_eps=relative_eps,
        scale=scale,
        center=center,
    )
    for x in seq:
        if det.update(x):
            break
    return {
        "stop_step": det.stop_step,
        "stopped": det.stop_step is not None,
        "phase": det.phase(),
        "reason": det.reason(),
        "effective_slope_eps": det.effective_slope_eps,
        "scale": det._scale,
        "center": det._center,
        "stats": det.stats(),
    }


if __name__ == "__main__":
    # Demo: compare the modes on small-scale data (mean ~1.0, amplitude ~0.05)
    import math as _m
    import random

    random.seed(0)
    warmup, t_conv = 500, 1500
    demo = []
    for t in range(5000):
        if t < warmup:
            base = 1.00
        else:
            base = 1.00 + 0.05 * (1.0 - _m.exp(-(t - warmup) / ((t_conv - warmup) / 4)))
            base = min(base, 1.05)
        demo.append(base + random.gauss(0, 0.10))   # large noise, small trend

    for mode in ("absolute", "relative", "normalized"):
        out = find_breakpoint(demo, warmup_steps=warmup, window_size=100, eps_mode=mode)
        print(f"eps_mode={mode:<10} stop_step={out['stop_step']}  "
              f"eff_slope_eps={out['effective_slope_eps']:.5f}")
