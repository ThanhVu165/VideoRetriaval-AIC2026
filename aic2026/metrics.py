from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(ranked_ids: Iterable[str], relevant_ids: set[str], k: int) -> float:
    """Binary recall for a single query over a ranked candidate list."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return float(bool(set(list(ranked_ids)[:k]) & relevant))


def mean_recall_at_k(all_ranked_ids: Iterable[Iterable[str]], all_relevant_ids: Iterable[set[str]], k: int) -> float:
    """Macro-average Recall@k across queries with explicit relevant IDs."""
    scores = [recall_at_k(ranked, relevant, k) for ranked, relevant in zip(all_ranked_ids, all_relevant_ids)]
    return sum(scores) / len(scores) if scores else 0.0


def reciprocal_rank(ranked_ids: Iterable[str], relevant_ids: set[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
