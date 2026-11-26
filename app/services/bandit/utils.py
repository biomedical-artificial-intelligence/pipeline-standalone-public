import numpy as np
import logging
from .algorithm import UCB, Bandit, OPTIMISTIC_UCB

logger = logging.getLogger(__name__)

aminoacid = "ARNDCEQGHILKMFPSTWYV"
aminoacid_len = len(aminoacid)
value_min = 0.5
value_max = 3.0


def normalize_score(s, min_val=value_min, max_val=value_max):
    s = max(min_val, min(max_val, s))
    return (s - min_val) / (max_val - min_val)


def index_to_char(index):
    return aminoacid[index]


def indice_to_string(indices):
    return "".join([index_to_char(i) for i in indices])


def char_to_index(char):
    return aminoacid.index(char)


def string_to_index(string):
    return [char_to_index(c) for c in string]


def cal_dock_score_matrix(pvc, length, batch_size):
    w_matrix = np.zeros((batch_size, aminoacid_len, length))
    scores = []
    rankings = []

    for j, seq_score_ranking in enumerate(pvc):
        if len(seq_score_ranking) == 3:
            seq, score, ranking = seq_score_ranking
            rankings.append(ranking)
        else:
            seq, score = seq_score_ranking
            rankings.append(None)

        seq_idx = string_to_index(seq)
        normalized_score = normalize_score(score)
        scores.append(normalized_score)

        for i, s in enumerate(seq_idx):
            w_matrix[j, s, i] = 1

    return scores, w_matrix, rankings


class ucb_config:
    pass


def init_ucb_param(length: int, iter: int = 0) -> UCB:
    uconf = ucb_config()
    uconf.N = length
    uconf.mus = np.zeros((aminoacid_len, uconf.N))
    # [FIX-v2 A] ucbs 낙관적 초기화(미시도 arm 우선 탐색). [FIX-v2 D] sigma_t 제거.
    uconf.ucbs = np.full((aminoacid_len, uconf.N), OPTIMISTIC_UCB)
    uconf.times = np.zeros((aminoacid_len, uconf.N))
    uconf.delta = 0.5
    uconf.alpha = np.ones(uconf.N)
    uconf.iter = iter
    return UCB(uconf)


def init_beta_param(length: int) -> Bandit:
    return Bandit(N=length)


def generate_random_sequence(length: int) -> str:
    return "".join(np.random.choice(list(aminoacid), length))