from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingWeights:
    retrieval: float = 0.65
    temporal: float = 0.20
    multimodal: float = 0.15


def _normalize(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 1e-12:
        return np.ones_like(x) if hi > 0 else np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rerank_candidates(candidates: pd.DataFrame, weights: RankingWeights | None = None) -> pd.DataFrame:
    """Fuse candidate-generation, temporal and multimodal evidence.

    The function deliberately keeps all component scores in the output so
    later learned-to-rank training can replace the fixed weighted sum.
    """
    if candidates.empty:
        return candidates.copy()
    w = weights or RankingWeights()
    out = candidates.copy()
    retrieval = _normalize(out["retrieval_score"])
    temporal = _normalize(out.get("temporal_score", pd.Series(np.zeros(len(out)))))
    multimodal = _normalize(out.get("multimodal_score", pd.Series(np.zeros(len(out)))))
    out["rank_score"] = w.retrieval * retrieval + w.temporal * temporal + w.multimodal * multimodal
    out["rank_score"] = out["rank_score"].astype(float)
    return out.sort_values("rank_score", ascending=False).reset_index(drop=True)


def top_k_submission(candidates: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """Return a deterministic top-k view without imposing an official schema."""
    if k <= 0:
        raise ValueError("k must be positive")
    if "rank_score" not in candidates.columns:
        raise ValueError("rank_score is required; call rerank_candidates first")
    return candidates.sort_values(
        ["rank_score", "video_id"], ascending=[False, True], kind="mergesort"
    ).head(k).reset_index(drop=True)
