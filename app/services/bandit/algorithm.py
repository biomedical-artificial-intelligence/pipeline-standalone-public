import numpy as np
from scipy.stats import beta, dirichlet
import logging

logger = logging.getLogger(__name__)

aminoacid_len = 20


class UCB:
    def __init__(self, config):
        self.mus = config.mus        # (20, N)
        self.ucbs = config.ucbs      # (20, N)
        self.sigma_t = config.sigma_t  # (20, N)
        self.N = config.N
        self.t = config.iter
        self.times = config.times    # (20, N)
        self.delta = config.delta
        self.alpha = config.alpha    # (N,)
        self.select = 0
        self.aa = 0

    def select_ac_one(self, seqsidx, *, advance_time: bool = True):
        action = np.array(seqsidx)

        if self.t < 500:
            self.select = int(np.random.choice(range(self.N), size=1)[0])
            self.aa = int(np.random.choice(range(self.mus.shape[0]), size=1)[0])
        else:
            prob_s = dirichlet(self.alpha).rvs()[0]
            self.select = int(np.random.choice(range(self.N), size=1, p=prob_s)[0])

            ucb_values = self.ucbs[:, self.select]
            max_indices = np.where(ucb_values == ucb_values.max())[0]
            self.aa = int(np.random.choice(max_indices, size=1)[0])

        action[self.select] = self.aa

        if advance_time:
            self.t += 1

        return action

    def _update_single_cell(self, aa: int, pos: int, outcome: float):
        sigma = 1

        self.alpha[pos] += 1 - outcome

        self.mus[aa, pos] = (
            self.mus[aa, pos] * self.times[aa, pos] + outcome
        ) / (self.times[aa, pos] + 1)

        self.sigma_t[aa, pos] = np.sqrt(
            1 / (1 / (self.sigma_t[aa, pos] ** 2) + 1 / (sigma ** 2))
        )

        self.times[aa, pos] += 1

        if self.times[aa, pos] > 0:
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
        outcome을 mutation 수로 나눠서 각 위치에 반영.
        """
        mutated_positions = [
            i for i, (s_aa, aa) in enumerate(zip(seed_idx, seqsidx))
            if s_aa != aa
        ]

        if not mutated_positions:
            return

        effective_outcome = float(outcome) / len(mutated_positions)
        for pos in mutated_positions:
            self._update_single_cell(int(seqsidx[pos]), int(pos), effective_outcome)


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