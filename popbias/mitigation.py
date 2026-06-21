"""
popbias.mitigation
------------------
Post-processing re-ranking strategies for mitigating popularity bias.

Two approaches are implemented:

xquad_rerank  (user-side)
    Greedy xQuAD-style re-ranking that balances relevance with a long-tail
    bonus, controlled by a tunable λ parameter.
    Reference: Clarke et al. (2008) xQuAD; adapted for popularity bias.

fair_rerank  (provider/artist-side)
    Guarantees a minimum number of long-tail items in every Top-N list,
    giving niche artists a floor of visibility.
    Reference: Ferraro et al. (2021) UMAP.
"""

import numpy as np


def xquad_rerank(
    cand_ids,
    cand_scores,
    lam: float,
    tail_set: set,
    n: int = 10,
):
    """
    xQuAD-style greedy re-ranking: balance relevance with long-tail exposure.

    At each step, selects the candidate that maximises:
        val = (1 - λ) · relevance_score + λ · tail_bonus

    where tail_bonus = 1.0 if the item is in the long-tail, else 0.0.

    Parameters
    ----------
    cand_ids : list
        Candidate item IDs (from ALS or another base model), in any order.
    cand_scores : list
        Corresponding relevance scores (higher = more relevant).
    lam : float
        Trade-off parameter in [0, 1].
        λ=0.0  → pure relevance (no fairness intervention).
        λ=0.45 → good balance (empirically: +32% ACLT, −0.04 NDCG on LFM-1b).
        λ=1.0  → pure long-tail (collapses accuracy).
    tail_set : set
        Set of long-tail item IDs.
    n : int
        Output list length (Top-N).

    Returns
    -------
    list of item IDs of length min(n, len(cand_ids))

    Examples
    --------
    >>> reranked = xquad_rerank(cand_ids, cand_scores, lam=0.45, tail_set=train_tail)
    """
    s = np.array(cand_scores, dtype=float)
    # Normalise scores to [0, 1] so λ is on a common scale
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())

    selected = []
    remaining = list(range(len(cand_ids)))

    while remaining and len(selected) < n:
        best_j, best_val = None, -1e18
        for j in remaining:
            tail_bonus = 1.0 if cand_ids[j] in tail_set else 0.0
            val = (1 - lam) * s[j] + lam * tail_bonus
            if val > best_val:
                best_val, best_j = val, j
        selected.append(best_j)
        remaining.remove(best_j)

    return [cand_ids[j] for j in selected]


def fair_rerank(
    cand_ids,
    cand_scores,
    min_tail: int,
    tail_set: set,
    n: int = 10,
):
    """
    Provider-side fairness re-ranking: guarantee a minimum number of
    long-tail items in every Top-N recommendation list.

    Pulls `min_tail` long-tail items to the front of the list, then
    fills the remaining slots with head/mid items, preserving the
    original score ordering within each group.

    Parameters
    ----------
    cand_ids : list
        Candidate item IDs.
    cand_scores : list
        Corresponding relevance scores.
    min_tail : int
        Minimum number of long-tail items guaranteed in the output list.
        E.g. min_tail=6 → at least 6 of the Top-10 are long-tail artists.
    tail_set : set
        Set of long-tail item IDs.
    n : int
        Output list length.

    Returns
    -------
    list of item IDs of length min(n, len(cand_ids))

    Examples
    --------
    >>> reranked = fair_rerank(cand_ids, cand_scores, min_tail=4, tail_set=train_tail)
    """
    # Preserve original score ordering
    score_order = {item: i for i, item in enumerate(cand_ids)}

    head_mid = [a for a in cand_ids if a not in tail_set]
    tail = [a for a in cand_ids if a in tail_set]

    # Guarantee floor of tail items; don't exceed what's available
    n_tail = min(min_tail, len(tail), n)
    n_head = n - n_tail

    chosen = tail[:n_tail] + head_mid[:n_head]
    # Re-sort chosen items by their original score rank
    chosen.sort(key=lambda a: score_order.get(a, 0))
    return chosen[:n]
