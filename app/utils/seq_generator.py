from __future__ import annotations

import random
from typing import List, Set, Optional, Tuple


AA20 = "ARNDCEQGHILKMFPSTWYV"  # bandit AMINOACID와 동일한 순서


def kmer_set(seq: str, k: int) -> Set[str]:
    if k <= 0:
        raise ValueError("k must be >= 1")
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def mutate_point(
    seq: str,
    n_mut: int,
    alphabet: str = AA20,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Apply n_mut point mutations at distinct positions.
    """
    rng = rng or random
    L = len(seq)
    if n_mut <= 0:
        return seq
    if n_mut > L:
        n_mut = L

    positions = rng.sample(range(L), k=n_mut)
    s = list(seq)

    for pos in positions:
        cur = s[pos]
        # choose a different AA
        choices = [a for a in alphabet if a != cur]
        s[pos] = rng.choice(choices)
    return "".join(s)


def mutate_segment_resample(
    seq: str,
    seg_len: int,
    alphabet: str = AA20,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Randomly pick a segment and resample it (like a small local randomization).
    """
    rng = rng or random
    L = len(seq)
    if seg_len <= 0:
        return seq
    seg_len = min(seg_len, L)
    start = rng.randrange(0, L - seg_len + 1)

    s = list(seq)
    for i in range(start, start + seg_len):
        s[i] = rng.choice(alphabet)
    return "".join(s)


def generate_mutant_candidates_from_seed(
    seed_seq: str,
    num_candidates: int,
    alphabet: str = AA20,
    seed: int = 0,
    # mutation settings
    point_mut_range: Tuple[int, int] = (1, 4),     # 1~4 point mutations
    seg_resample_prob: float = 0.15,               # 15% chance to also do segment resample
    seg_len_range: Tuple[int, int] = (3, 6),       # resample 3~6 aa segment
    max_tries: int = 2_000_000,
    include_seed: bool = False,
) -> List[str]:
    """
    Generate unique mutant candidates derived from a given seed sequence.
    """
    rng = random.Random(seed)
    seen = set()
    out: List[str] = []

    if include_seed:
        seen.add(seed_seq)
        out.append(seed_seq)

    tries = 0
    lo, hi = point_mut_range
    if lo < 0 or hi < 0 or lo > hi:
        raise ValueError("Invalid point_mut_range")

    seg_lo, seg_hi = seg_len_range
    if seg_lo < 0 or seg_hi < 0 or seg_lo > seg_hi:
        raise ValueError("Invalid seg_len_range")

    while len(out) < num_candidates and tries < max_tries:
        tries += 1

        # 1) point mutate
        n_mut = rng.randint(lo, hi)
        cand = mutate_point(seed_seq, n_mut=n_mut, alphabet=alphabet, rng=rng)

        # 2) optional segment resample
        if rng.random() < seg_resample_prob:
            seg_len = rng.randint(seg_lo, seg_hi)
            cand = mutate_segment_resample(cand, seg_len=seg_len, alphabet=alphabet, rng=rng)

        if cand in seen:
            continue
        seen.add(cand)
        out.append(cand)

    if len(out) < num_candidates:
        raise RuntimeError(
            f"Could not generate enough unique mutant candidates: "
            f"got {len(out)} / {num_candidates} (tries={tries}). "
            f"Consider increasing max_tries or loosening constraints."
        )
    return out


def greedy_maxmin_diverse_selection(
    candidates: List[str],
    select_k: int,
    kmer_k: int = 3,
    seed: int = 0,
) -> List[str]:
    """
    Greedy max-min selection to maximize diversity.
    Distance = Jaccard distance over k-mer sets.
    """
    if select_k <= 0:
        return []
    if select_k > len(candidates):
        raise ValueError("select_k cannot exceed number of candidates")

    rng = random.Random(seed)
    kmers = [kmer_set(s, kmer_k) for s in candidates]

    # start from a random candidate
    start_idx = rng.randrange(len(candidates))
    selected_idxs = [start_idx]

    min_dists = [jaccard_distance(kmers[i], kmers[start_idx]) for i in range(len(candidates))]
    min_dists[start_idx] = -1.0

    while len(selected_idxs) < select_k:
        best_idx = max(range(len(candidates)), key=lambda i: min_dists[i])
        selected_idxs.append(best_idx)

        for i in range(len(candidates)):
            if min_dists[i] < 0:
                continue
            d = jaccard_distance(kmers[i], kmers[best_idx])
            if d < min_dists[i]:
                min_dists[i] = d
        min_dists[best_idx] = -1.0

    return [candidates[i] for i in selected_idxs]


def build_initial_queue_sequences_from_seed(
    seed_seq: str,
    N_workers: int,
    candidate_multiplier: int = 10,   # generate 10N mutant candidates
    queue_multiplier: int = 3,       # choose 3N for initial queue
    kmer_k: int = 3,
    seed: int = 0,
    # mutation knobs
    point_mut_range: Tuple[int, int] = (1, 4),
    seg_resample_prob: float = 0.15,
    seg_len_range: Tuple[int, int] = (3, 6),
    include_seed: bool = False,
) -> List[str]:
    """
    Main entry:
      - generate mutant candidates derived from seed sequence
      - pick 3N diverse sequences
    """
    if N_workers <= 0:
        raise ValueError("N_workers must be > 0")

    num_candidates = candidate_multiplier * N_workers
    num_queue = queue_multiplier * N_workers

    candidates = generate_mutant_candidates_from_seed(
        seed_seq=seed_seq,
        num_candidates=num_candidates,
        seed=seed,
        point_mut_range=point_mut_range,
        seg_resample_prob=seg_resample_prob,
        seg_len_range=seg_len_range,
        include_seed=include_seed,
    )

    selected = greedy_maxmin_diverse_selection(
        candidates=candidates,
        select_k=num_queue,
        kmer_k=kmer_k,
        seed=seed,
    )
    return selected


if __name__ == "__main__":
    # Example
    seed_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"  # 30-mer 예시
    N = 20

    init_seqs = build_initial_queue_sequences_from_seed(
        seed_seq=seed_seq,
        N_workers=N,
        candidate_multiplier=10,  # 10N 후보 생성
        queue_multiplier=3,       # 3N 선택
        kmer_k=3,
        seed=42,
        point_mut_range=(1, 4),   # 1~4 point mutations
        seg_resample_prob=0.10,   # 10% 확률로 segment resample
        seg_len_range=(3, 6),
        include_seed=False,
    )

    print(f"Selected {len(init_seqs)} sequences for initial queue (expected {3*N})")
    for s in init_seqs[:10]:
        print(s)

    # 여기서 init_seqs를 dock_queue에 push 하시면 됩니다.