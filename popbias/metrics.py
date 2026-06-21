"""
popbias.metrics
---------------
Core popularity-bias measurement functions.

All functions are dataset-agnostic: they operate on plain Python dicts
and lists so they can be dropped into any RecSys evaluation pipeline.

Metrics implemented
~~~~~~~~~~~~~~~~~~~
- gini              : Gini coefficient of item exposure / play counts
- popularity_groups : Partition items into head / mid / tail buckets
- arp               : Average Recommendation Popularity (Abdollahpouri 2019)
- aclt              : Average Coverage of Long-Tail items  (Abdollahpouri 2019)
- catalog_coverage  : Fraction of the catalog that appears in any recommendation
- rec_metrics       : Convenience wrapper returning all four metrics at once
"""

import numpy as np


def gini(x) -> float:
    """
    Gini coefficient of an array of counts (plays, exposures, etc.).

    Returns 0.0 for perfect equality (every item equally exposed) and
    approaches 1.0 for a monopoly (one item gets everything).

    Parameters
    ----------
    x : array-like of non-negative numbers

    Returns
    -------
    float in [0, 1]

    References
    ----------
    Celma, O. (2010). *Music Recommendation and Discovery*. Springer.
    """
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def popularity_groups(item_plays: dict, head: float = 0.20, mid: float = 0.40) -> dict:
    """
    Partition items into popularity tiers based on their rank percentile.

    Parameters
    ----------
    item_plays : dict
        Mapping of item_id -> play/interaction count.
    head : float
        Rank-percentile threshold for the *head* tier (default 0.20 = top 20%).
    mid : float
        Rank-percentile threshold for the *mid* tier (default 0.40).
        Items above `mid` rank-percentile are classified as *tail*.

    Returns
    -------
    dict
        Mapping of item_id -> 'head' | 'mid' | 'tail'.

    Examples
    --------
    >>> groups = popularity_groups({"a": 1000, "b": 500, "c": 10, "d": 5})
    >>> groups["a"]
    'head'
    """
    order = sorted(item_plays, key=item_plays.get, reverse=True)
    n = len(order)
    out = {}
    for rank, item in enumerate(order):
        rp = rank / n
        if rp <= head:
            out[item] = "head"
        elif rp <= mid:
            out[item] = "mid"
        else:
            out[item] = "tail"
    return out


def arp(rec_lists, item_plays: dict) -> float:
    """
    Average Recommendation Popularity (ARP).

    Measures how mainstream the average recommended item is across all users.
    Higher ARP means the recommender is biased toward already-popular items.

    Parameters
    ----------
    rec_lists : iterable of lists
        Each element is a Top-N recommendation list of item IDs for one user.
    item_plays : dict
        Mapping of item_id -> global play / interaction count (training set).

    Returns
    -------
    float
        Mean popularity of recommended items averaged over all users.

    References
    ----------
    Abdollahpouri et al. (2019). *Managing Popularity Bias in Recommender
    Systems with Personal Expected Popularity*. FLAIRS 2019.
    """
    vals = [
        np.mean([item_plays.get(i, 0) for i in items])
        for items in rec_lists
        if items
    ]
    return float(np.mean(vals)) if vals else 0.0


def aclt(rec_lists, tail_set: set) -> float:
    """
    Average Coverage of Long-Tail items (ACLT).

    Measures what fraction of each user's recommendations come from the
    long-tail (niche) item set. Higher ACLT means less popularity bias.

    Parameters
    ----------
    rec_lists : iterable of lists
        Each element is a Top-N recommendation list of item IDs for one user.
    tail_set : set
        Set of item IDs that belong to the long-tail (e.g. bottom 80% by plays).

    Returns
    -------
    float in [0, 1]

    References
    ----------
    Abdollahpouri et al. (2019). FLAIRS 2019.
    """
    vals = [
        np.mean([1.0 if i in tail_set else 0.0 for i in items])
        for items in rec_lists
        if items
    ]
    return float(np.mean(vals)) if vals else 0.0


def catalog_coverage(rec_lists, n_items: int) -> float:
    """
    Fraction of the full item catalog that appears in at least one
    recommendation list.

    Parameters
    ----------
    rec_lists : iterable of lists
        Recommendation lists for all users.
    n_items : int
        Total number of items in the catalog.

    Returns
    -------
    float in [0, 1]
    """
    seen: set = set()
    for items in rec_lists:
        seen.update(items)
    return len(seen) / n_items if n_items > 0 else 0.0


def rec_metrics(
    recs: dict,
    item_plays: dict,
    tail_set: set,
    n_items: int,
) -> dict:
    """
    Convenience wrapper: compute ARP, ACLT, Gini, and Catalog Coverage
    in a single call.

    Parameters
    ----------
    recs : dict
        Mapping of user_id -> list of recommended item IDs.
    item_plays : dict
        Mapping of item_id -> global play count (training set only).
    tail_set : set
        Set of long-tail item IDs.
    n_items : int
        Total catalog size.

    Returns
    -------
    dict with keys: ARP, ACLT, Exposure_Gini, Catalog_Coverage

    Examples
    --------
    >>> metrics = rec_metrics(recs_als, train_item_plays, train_tail, n_items)
    >>> print(metrics)
    {'ARP': 15215.5, 'ACLT': 0.0, 'Exposure_Gini': 0.996, 'Catalog_Coverage': 0.011}
    """
    arps, aclts = [], []
    exposure = np.zeros(n_items, dtype=float)

    for items in recs.values():
        if not items:
            continue
        arps.append(np.mean([item_plays.get(i, 0) for i in items]))
        aclts.append(np.mean([1.0 if i in tail_set else 0.0 for i in items]))
        for i in items:
            if 0 <= i < n_items:
                exposure[i] += 1

    return {
        "ARP": float(np.mean(arps)) if arps else 0.0,
        "ACLT": float(np.mean(aclts)) if aclts else 0.0,
        "Exposure_Gini": gini(exposure),
        "Catalog_Coverage": float((exposure > 0).sum() / n_items) if n_items else 0.0,
    }
