import numpy as np
from scipy.stats import beta, dirichlet
import logging

logger = logging.getLogger(__name__)

aminoacid_len = 20

# [FIX-v2 A] 미시도 arm을 먼저 탐색하게 만드는 낙관적 UCB 초기값.
#   mus∈[0,1], bonus≥0 이므로 방문 셀의 ucb는 항상 < OPTIMISTIC_UCB 이어야 함.
OPTIMISTIC_UCB = 1e3
# [FIX-v2 B] Dirichlet concentration(=alpha 합) 상한. 위치 선호 비율은 보존하되
#   분산이 0으로 수렴해 위치 선택이 결정론적으로 굳는 것을 방지(장기 실행 안전장치).
#   cap = ALPHA_CAP_MULT * N
ALPHA_CAP_MULT = 50.0


class UCB:
    def __init__(self, config):
        self.mus = config.mus        # (20, N)
        self.ucbs = config.ucbs      # (20, N)
        # [FIX-v2 D] sigma_t 제거(미사용 죽은 코드)
        self.N = config.N
        self.t = config.iter
        self.times = config.times    # (20, N)
        self.delta = config.delta
        self.alpha = config.alpha    # (N,)
        # [FIX-v2 B] alpha 합 상한
        self.alpha_cap = ALPHA_CAP_MULT * self.N
        # [FIX-v2 E] 랜덤 탐색 구간 길이(매직상수 500 제거 → 10*N).
        #   이 step 전까지는 무작위 탐색, 이후 UCB 가이드 선택으로 전환.
        self.random_phase_steps = 10 * self.N
        self.select = 0
        self.aa = 0

    def select_ac_one(self, seqsidx, *, advance_time: bool = True):
        action = np.array(seqsidx)

        if self.t < self.random_phase_steps:
            self.select = int(np.random.choice(range(self.N), size=1)[0])
            self.aa = int(np.random.choice(range(self.mus.shape[0]), size=1)[0])
        else:
            prob_s = dirichlet(self.alpha).rvs()[0]
            self.select = int(np.random.choice(range(self.N), size=1, p=prob_s)[0])

            # [FIX-v2 A] untried-first: 이 위치에서 아직 안 써본 아미노산이 있으면
            #   그것을 먼저 시도(표준 UCB의 낙관적 탐색). 없으면 argmax UCB.
            #   단, 현재 base 잔기는 제외 — 그걸 고르면 변이 없음(no-op)이 되어
            #   base와 동일한 중복 후보가 생성되므로.
            base_aa = int(action[self.select])
            untried = np.where(self.times[:, self.select] == 0)[0]
            untried = untried[untried != base_aa]
            if untried.size > 0:
                self.aa = int(np.random.choice(untried))
            else:
                ucb_values = self.ucbs[:, self.select]
                max_indices = np.where(ucb_values == ucb_values.max())[0]
                self.aa = int(np.random.choice(max_indices, size=1)[0])

        action[self.select] = self.aa

        if advance_time:
            self.t += 1

        return action

    def _update_single_cell(self, aa: int, pos: int, outcome: float):
        # 나쁜 outcome → 해당 위치 alpha↑ → 더 자주 변이(exploration 강화)
        self.alpha[pos] += 1 - outcome
        # [FIX-v2 B] concentration 상한: 합이 cap을 넘으면 비례 축소(상대 선호 보존,
        #            Dirichlet 분산이 0으로 수렴하는 붕괴 방지)
        total = self.alpha.sum()
        if total > self.alpha_cap:
            self.alpha *= (self.alpha_cap / total)

        self.mus[aa, pos] = (
            self.mus[aa, pos] * self.times[aa, pos] + outcome
        ) / (self.times[aa, pos] + 1)

        # [FIX-v2 D] sigma_t(미사용) 업데이트 제거

        self.times[aa, pos] += 1

        # UCB1 bonus
        log_val = np.log(max(self.N * self.t ** 2 / self.delta, 1.0))
        exploration = max(2 * log_val / self.times[aa, pos], 0.0)
        self.ucbs[aa, pos] = self.mus[aa, pos] + np.sqrt(exploration)

    def update(self, outcome):
        """Search phase: UCB가 선택한 (select, aa) 위치 업데이트"""
        self._update_single_cell(int(self.aa), int(self.select), float(outcome))

    def update_from_sequence(self, outcome, seqsidx):
        """Warm-up seed 전체 서열 기반 업데이트"""
        for pos, aa in enumerate(seqsidx):
            self._update_single_cell(int(aa), int(pos), float(outcome))

    def update_from_mutations(self, outcome, seed_idx, seqsidx):
        """
        Warm-up 초기 3N 후보: seed 대비 mutation 위치만 업데이트.
        [FIX-v2 C] outcome을 mutation 수로 나누지 않고 각 변이 위치에 전체 outcome
        반영(다중변이 후보 셀의 mus 과소추정 제거).
        """
        mutated_positions = [
            i for i, (s_aa, aa) in enumerate(zip(seed_idx, seqsidx))
            if s_aa != aa
        ]

        if not mutated_positions:
            return

        for pos in mutated_positions:
            self._update_single_cell(int(seqsidx[pos]), int(pos), float(outcome))


class Bandit:
    def __init__(self, N):
        self.N = N
        self.A = np.ones((aminoacid_len, self.N))
        self.B = np.ones((aminoacid_len, self.N))
        self.times = np.zeros(self.N)
        self.alpha = np.ones(self.N)
        self.t = 1
        self.select = 1
        self.AA = 0

    def __repr__(self) -> str:
        return f"Bandit(N={self.N})"

    def select_ac_one(self, seqsidx):
        prob_s = dirichlet(self.alpha).rvs()[0]
        self.select = int(np.random.choice(range(len(self.alpha)), size=1, p=prob_s)[0])

        if self.t < 50 * self.N:
            self.action = np.random.choice(range(aminoacid_len), size=self.N)
        else:
            prob_M = beta(self.A[:, self.select], self.B[:, self.select]).rvs()
            self.AA = int(np.argmax(prob_M))

        self.action = np.array(seqsidx)
        self.action[self.select] = self.AA
        self.t += 1
        return self.action

    def update(self, outcome, w_matrix, alpha):
        self.alpha[self.select] += 1 - outcome
        self.A[:, self.select] += alpha * outcome * w_matrix[:, self.select]
        self.B[:, self.select] += alpha * (1 - outcome) * w_matrix[:, self.select]
