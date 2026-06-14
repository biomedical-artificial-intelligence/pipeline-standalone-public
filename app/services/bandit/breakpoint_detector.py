"""
breakpoint_detector.py

UCB 기반 ZDock 시뮬레이션용 '조기 종료(중단점) 탐지' 로직만 분리한 독립 모듈.
시뮬레이션/시각화 코드와 무관하게 실제 탐색 루프에 바로 끼워 쓸 수 있다.

구성:
  - core 판단기      : early_stopping.EarlyStopper (수정하지 않는 원본 로직)
  - 본 모듈이 추가    : 초기 랜덤 탐색 구간 제외(warm-up) + 온라인/오프라인 API

의존성: 표준 라이브러리만 사용 (numpy 불필요).

────────────────────────────────────────────────────────────────────
규약
  - 스코어는 '높을수록 좋음' (ZDock 스코어, EarlyStopper의 best_avg 최대 추적과 일치).
  - warm-up: UCB 초기 N step은 추세 없는 랜덤 탐색 구간이므로 중단점 탐색에서 제외.
             해당 구간 점수는 EarlyStopper에 주입하지 않아 rolling avg/patience/slope를
             오염시키지 않는다.

사용 (온라인 — 실제 탐색 루프):
    from breakpoint_detector import BreakpointDetector

    det = BreakpointDetector(window_size=100, warmup_steps=500)
    for step in range(max_steps):
        score = run_ucb_zdock_step(...)     # 한 스텝 평가 (느린 계산)
        if det.update(score):               # True면 수렴으로 판단 → 종료
            print("조기 종료:", det.reason())
            break

사용 (오프라인 — 이미 가진 점수 시퀀스):
    from breakpoint_detector import find_breakpoint
    out = find_breakpoint(scores, warmup_steps=500, window_size=100)
    print(out["stop_step"], out["reason"])
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Iterable

from .early_stopping import EarlyStopper


class BreakpointDetector:
    """
    EarlyStopper에 warm-up(초기 랜덤 탐색 제외)을 더한 중단점 탐지기.

    파라미터
        window_size      (W)  : rolling avg/std 윈도우 크기            (기본 100)
        slope_horizon    (SH) : 기울기 비교 거리 배수. H = SH × W      (기본 3)
        slope_eps        (ε)  : 의미 있는 개선의 최소값                (기본 0.01)
        patience_windows (PW) : patience 허용 배수. P = PW × W         (기본 5)
        buffer_windows   (BW) : 안전 버퍼 기본 길이 배수. buf = BW × W (기본 2)
        adaptive_buffer       : rolling_std 기반 버퍼 동적 연장        (기본 True)
        warmup_steps     (N)  : 초기 랜덤 탐색 제외 구간 길이          (기본 500)
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
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = warmup_steps
        self._stopper = EarlyStopper(
            window_size=window_size,
            slope_horizon=slope_horizon,
            slope_eps=slope_eps,
            patience_windows=patience_windows,
            buffer_windows=buffer_windows,
            adaptive_buffer=adaptive_buffer,
        )
        self._step = 0                 # 지금까지 입력된 총 스텝 수(절대 인덱스 기준)
        self._stop_step: int | None = None

    # ──────────────────────────────────────────────────────────────
    # 온라인 API
    # ──────────────────────────────────────────────────────────────
    def update(self, score: float) -> bool:
        """
        스코어 1개를 입력하고 '지금 종료해야 하는가'를 반환한다.
        warm-up 구간(첫 N 스텝)의 점수는 탐지에 사용하지 않고 항상 False를 반환.
        """
        i = self._step
        self._step += 1

        if i < self.warmup_steps:
            return False               # 랜덤 탐색 구간 → 탐지 제외

        self._stopper.update(float(score))
        stop = self._stopper.should_stop()
        if stop and self._stop_step is None:
            self._stop_step = i        # 절대 스텝 인덱스로 기록
        return stop

    def should_stop(self) -> bool:
        """현재 종료 상태인지."""
        return self._stopper.should_stop()

    def phase(self) -> str:
        """'MONITORING' | 'BUFFER' | 'STOPPED' (warm-up 중에도 MONITORING)."""
        return self._stopper.phase()

    def reason(self) -> str:
        """종료 사유 문자열(미종료면 빈 문자열)."""
        return self._stopper.reason()

    @property
    def stop_step(self) -> int | None:
        """종료가 확정된 절대 스텝 인덱스(미종료면 None)."""
        return self._stop_step

    @property
    def n_steps(self) -> int:
        """지금까지 입력된 총 스텝 수."""
        return self._step

    def stats(self) -> dict:
        """EarlyStopper 내부 통계 + warm-up/스텝 정보."""
        s = dict(self._stopper.stats())
        s["warmup_steps"] = self.warmup_steps
        s["total_steps"] = self._step
        s["stop_step"] = self._stop_step
        return s


def find_breakpoint(
    scores: Iterable[float],
    *,
    warmup_steps: int = 500,
    window_size: int = 100,
    slope_horizon: int = 3,
    slope_eps: float = 0.01,
    patience_windows: int = 5,
    buffer_windows: int = 2,
    adaptive_buffer: bool = True,
) -> dict:
    """
    이미 가진 점수 시퀀스에서 중단점을 한 번에 찾는 오프라인 헬퍼.
    종료가 확정되면 즉시 순회를 멈춘다(이후 점수는 보지 않음).

    반환 dict:
        stop_step : 종료 확정 절대 스텝(없으면 None)
        stopped   : 종료 여부(bool)
        phase     : 최종 상태 문자열
        reason    : 종료 사유 문자열
        stats     : 상세 통계 dict
    """
    det = BreakpointDetector(
        window_size=window_size,
        slope_horizon=slope_horizon,
        slope_eps=slope_eps,
        patience_windows=patience_windows,
        buffer_windows=buffer_windows,
        adaptive_buffer=adaptive_buffer,
        warmup_steps=warmup_steps,
    )
    for s in scores:
        if det.update(s):
            break
    return {
        "stop_step": det.stop_step,
        "stopped": det.stop_step is not None,
        "phase": det.phase(),
        "reason": det.reason(),
        "stats": det.stats(),
    }


if __name__ == "__main__":
    # 간단한 데모: 상승 후 평탄(plateau) 시퀀스에서 중단점 탐지
    import math
    import random

    random.seed(0)
    warmup, t_conv = 500, 1500
    demo = []
    for t in range(5000):
        if t < warmup:                       # 랜덤 탐색 구간
            base = 8.0
        else:                                # 상승→평탄
            base = 8.0 + 7.0 * (1.0 - math.exp(-(t - warmup) / ((t_conv - warmup) / 4)))
            base = min(base, 8.0 + 7.0)
        demo.append(base + random.gauss(0, 0.3))

    out = find_breakpoint(demo, warmup_steps=warmup, window_size=100)
    print("stop_step:", out["stop_step"])
    print("reason   :", out["reason"])
