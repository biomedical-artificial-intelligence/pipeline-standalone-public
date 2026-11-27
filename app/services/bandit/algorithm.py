import numpy as np
from scipy.stats import beta, dirichlet
import logging

logger = logging.getLogger(__name__)

aminoacid_len = 20

# [FIX-v2 A] Optimistic initial UCB so untried arms are explored first.
#   mus∈[0,1] and bonus≥0, so a visited cell's ucb must always be < OPTIMISTIC_UCB.
OPTIMISTIC_UCB = 1e3
# [FIX-v2 B] Cap on the Dirichlet concentration (= alpha sum). Keeps the relative
#   position preference while stopping the variance from collapsing to 0, which would
#   freeze position selection (a long-run safeguard).
#   cap = ALPHA_CAP_MULT * N
ALPHA_CAP_MULT = 50.0


class UCB:
    def __init__(self, config):
        self.mus = config.mus        # (20, N)
        self.ucbs = config.ucbs      # (20, N)
        # [FIX-v2 D] sigma_t removed (unused dead code)
        self.N = config.N
        self.t = config.iter
        self.times = config.times    # (20, N)
        self.delta = config.delta
        self.alpha = config.alpha    # (N,)
        # [FIX-v2 B] cap on the alpha sum
        self.alpha_cap = ALPHA_CAP_MULT * self.N
        # [FIX-v2 E] Length of the random-search phase (magic constant 500 → 10*N).
        #   Random until this step, UCB-guided selection afterwards.
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

            # [FIX-v2 A] untried-first: if this position still has amino acids that
            #   were never tried, try one of those first (standard optimistic UCB);
            #   otherwise take argmax UCB. The current base residue is excluded, since
            #   picking it is a no-op and would duplicate the base sequence.
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
        # Bad outcome → raise alpha at this position → mutate it more often (more exploration)
        self.alpha[pos] += 1 - outcome
        # [FIX-v2 B] concentration cap: scale down proportionally past the cap, which
        #            keeps the relative preference and avoids a variance collapse to 0
        total = self.alpha.sum()
        if total > self.alpha_cap:
            self.alpha *= (self.alpha_cap / total)

        self.mus[aa, pos] = (
            self.mus[aa, pos] * self.times[aa, pos] + outcome
        ) / (self.times[aa, pos] + 1)

        # [FIX-v2 D] sigma_t update removed (unused)

        self.times[aa, pos] += 1

        # UCB1 bonus
        log_val = np.log(max(self.N * self.t ** 2 / self.delta, 1.0))
        exploration = max(2 * log_val / self.times[aa, pos], 0.0)
        self.ucbs[aa, pos] = self.mus[aa, pos] + np.sqrt(exploration)

    def update(self, outcome):
        """Search phase: update the (select, aa) cell chosen by UCB."""
        self._update_single_cell(int(self.aa), int(self.select), float(outcome))

    def update_from_sequence(self, outcome, seqsidx):
        """Warm-up seed: update from the whole sequence."""
        for pos, aa in enumerate(seqsidx):
            self._update_single_cell(int(aa), int(pos), float(outcome))

    def update_from_mutations(self, outcome, seed_idx, seqsidx):
        """
        Warm-up initial 3N candidates: update only the positions mutated from the seed.
        [FIX-v2 C] the outcome is not divided by the mutation count; every mutated
        position takes the full outcome (no mus underestimation for multi-mutation cells).
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
