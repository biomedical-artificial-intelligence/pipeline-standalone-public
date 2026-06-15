"""
breakpoint_detector.py

UCB 기반 ZDock 시뮬레이션용 '조기 종료(중단점) 탐지' 로직만 분리한 독립 모듈.
시뮬레이션/시각화 코드와 무관하게 실제 탐색 루프에 바로 끼워 쓸 수 있다.

구성:
  - core 판단기      : early_stopping.EarlyStopper (수정하지 않는 원본 로직)
  - 본 모듈이 추가    : (1) 초기 랜덤 탐색 구간 제외(warm-up)
                        (2) 스코어 스케일에 둔감한 임계 모드 (absolute / relative / normalized)

의존성: 표준 라이브러리만 사용 (numpy 불필요).

────────────────────────────────────────────────────────────────────
임계(slope_eps) 스케일 문제
  EarlyStopper의 slope_eps는 '이동평균의 기울기(절대값)'에 대한 임계다.
  스코어 스케일이 데이터마다 다르면(예: 합성 진폭 ~10  vs  실측 ZDock 평균 ~1.0,
  이동평균 변동폭 ~0.05) 동일한 slope_eps가 한쪽에선 적정, 다른 쪽에선 과도하게 커
  '정체'로 오판 → 조기 종료가 트리거된다.

  이를 해결하기 위해 eps_mode 옵션 제공:
    - "absolute"   : slope_eps를 그대로 사용 (기존 동작, 기본값)
    - "relative"   : slope_eps를 데이터 스케일에 비례시켜 자동 산출
                     eff_eps = relative_eps × (scale / sqrt(window_size))
                     scale = 스코어 표준편차(warm-up 또는 명시값)
    - "normalized" : 스코어를 (x - center) / scale 로 정규화해 판단기에 주입
                     → slope_eps가 스케일 불변(정규화 표준편차 ≈ 1 기준)

  scale/center는 warm-up 구간 샘플로 자동 보정하거나(online),
  scale=/center= 인자로 직접 지정할 수 있다(offline: 전체 배열 통계 사용 권장).
────────────────────────────────────────────────────────────────────

규약
  - 스코어는 '높을수록 좋음' (ZDock 스코어, best_avg 최대 추적과 일치).
  - warm-up: UCB 초기 N step은 추세 없는 랜덤 탐색 구간이므로 중단점 탐색에서 제외.

사용 (온라인 — 실측처럼 스케일이 작은 데이터: normalized 권장):
    from breakpoint_detector import BreakpointDetector
    det = BreakpointDetector(window_size=100, warmup_steps=500,
                             eps_mode="normalized")     # warm-up으로 자동 스케일 보정
    for step in range(max_steps):
        score = run_ucb_zdock_step(...)
        if det.update(score):
            print("조기 종료:", det.reason()); break

사용 (오프라인):
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
    EarlyStopper에 warm-up(초기 랜덤 탐색 제외) + 스케일 둔감 임계 모드를 더한 중단점 탐지기.

    파라미터
        window_size      (W)  : rolling avg/std 윈도우 크기            (기본 100)
        slope_horizon    (SH) : 기울기 비교 거리 배수. H = SH × W      (기본 3)
        slope_eps        (ε)  : 의미 있는 개선의 최소값                (기본 0.01)
                                · eps_mode="absolute"  → 그대로 사용
                                · eps_mode="normalized"→ 정규화(std≈1) 스케일에서의 ε
        patience_windows (PW) : patience 허용 배수. P = PW × W         (기본 5)
        buffer_windows   (BW) : 안전 버퍼 기본 길이 배수. buf = BW × W (기본 2)
        adaptive_buffer       : rolling_std 기반 버퍼 동적 연장        (기본 True)
        warmup_steps     (N)  : 초기 랜덤 탐색 제외 구간 길이          (기본 500)

      스케일 둔감 옵션:
        eps_mode              : "absolute" | "relative" | "normalized"  (기본 "absolute")
        relative_eps          : eps_mode="relative"일 때 비례계수        (기본 0.1)
        scale                 : 스케일(표준편차) 직접 지정. None이면 warm-up으로 보정
        center                : 중심(평균) 직접 지정. None이면 warm-up으로 보정
        min_calibration       : warm-up 보정에 필요한 최소 샘플 수       (기본 30)
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

        # EarlyStopper 생성 인자(slope_eps는 모드에 따라 보정 후 확정)
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
        self._scale = 1.0           # 정규화/상대임계에 쓰는 데이터 스케일
        self._center = 0.0
        self.effective_slope_eps = slope_eps

        self._warm: list[float] = []   # warm-up 보정용 raw 샘플
        self._step = 0
        self._stop_step: Optional[int] = None

        # absolute 모드는 보정이 필요 없으므로 즉시 판단기 생성.
        # relative/normalized 모드는 warm-up 종료 시점에 보정 후 생성(지연).
        self._stopper: Optional[EarlyStopper] = None
        if eps_mode == ABSOLUTE and scale is None and center is None:
            self._stopper = EarlyStopper(**self._es_kwargs)
        # scale/center가 명시되면 absolute에서도 변환만 적용(보통 불필요)

    # ──────────────────────────────────────────────────────────────
    # 내부: 스케일 보정 + 판단기 생성
    # ──────────────────────────────────────────────────────────────
    def _calibrate_and_build(self) -> None:
        """warm-up 종료 시점(또는 즉시) 1회 호출. scale/center 확정 후 EarlyStopper 생성."""
        # 스케일/중심 결정: 명시값 우선, 없으면 warm-up 샘플로 추정
        if self._user_scale is not None:
            scale = float(self._user_scale)
            center = float(self._user_center) if self._user_center is not None else 0.0
        elif len(self._warm) >= self.min_calibration:
            center = fmean(self._warm)
            scale = pstdev(self._warm)
        else:
            # 보정 표본 부족 → absolute로 안전하게 폴백
            logger.warning(
                "[BreakpointDetector] 보정 표본 부족(%d < %d) → eps_mode=absolute로 폴백",
                len(self._warm), self.min_calibration,
            )
            self.eps_mode = ABSOLUTE
            center, scale = 0.0, 1.0

        if not scale or scale <= 0 or math.isnan(scale):
            scale = 1.0   # 분산 0 등 예외 방어

        self._center, self._scale = center, scale
        kw = dict(self._es_kwargs)

        if self.eps_mode == RELATIVE:
            # 이동평균 기울기의 자연 단위 ≈ scale / sqrt(W)
            kw["slope_eps"] = self.relative_eps * scale / math.sqrt(self._window_size)
        elif self.eps_mode == NORMALIZED:
            # 정규화(std≈1) 스케일에서 base slope_eps 그대로 적용
            kw["slope_eps"] = self._base_slope_eps
        # absolute: base 그대로

        self.effective_slope_eps = kw["slope_eps"]
        self._stopper = EarlyStopper(**kw)

    def _transform(self, score: float) -> float:
        """normalized 모드에서만 (x - center)/scale 적용. 그 외엔 원값."""
        if self.eps_mode == NORMALIZED:
            return (score - self._center) / self._scale
        return score

    # ──────────────────────────────────────────────────────────────
    # 온라인 API
    # ──────────────────────────────────────────────────────────────
    def update(self, score: float) -> bool:
        """
        스코어 1개를 입력하고 '지금 종료해야 하는가'를 반환한다.
        warm-up 구간(첫 N 스텝)의 점수는 탐지에 사용하지 않고 항상 False를 반환하되,
        relative/normalized 모드에선 이 구간 표본으로 스케일을 보정한다.
        """
        i = self._step
        self._step += 1
        score = float(score)

        if i < self.warmup_steps:
            if self.eps_mode != ABSOLUTE and self._user_scale is None:
                self._warm.append(score)     # 스케일 보정용 수집
            return False

        # 첫 탐지 스텝에서 판단기 미생성 상태면 보정 후 생성
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
        """종료가 확정된 절대 스텝 인덱스(미종료면 None)."""
        return self._stop_step

    @property
    def n_steps(self) -> int:
        return self._step

    def stats(self) -> dict:
        """EarlyStopper 내부 통계 + warm-up/스케일/임계 정보."""
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
    이미 가진 점수 시퀀스에서 중단점을 한 번에 찾는 오프라인 헬퍼.
    종료가 확정되면 즉시 순회를 멈춘다.

    calibrate:
        "auto"/"global" : relative/normalized 모드에서 scale/center를 '전체 배열' 통계로
                          계산해 사용(견고, 오프라인 기본). scale=/center=가 주어지면 그 값 우선.
        "warmup"        : warm-up 구간 표본으로만 보정(온라인과 동일 동작).

    반환 dict: stop_step, stopped, phase, reason, effective_slope_eps, scale, center, stats
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
    # 데모: 스케일이 작은(평균~1.0, 진폭~0.05) 데이터에서 모드별 거동 비교
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
        demo.append(base + random.gauss(0, 0.10))   # 진폭이 큰 노이즈, 작은 추세

    for mode in ("absolute", "relative", "normalized"):
        out = find_breakpoint(demo, warmup_steps=warmup, window_size=100, eps_mode=mode)
        print(f"eps_mode={mode:<10} stop_step={out['stop_step']}  "
              f"eff_slope_eps={out['effective_slope_eps']:.5f}")
