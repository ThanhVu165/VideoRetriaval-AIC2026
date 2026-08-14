from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "retrieval_score",
    "retrieval_best_score",
    "retrieval_topk_mean",
    "retrieval_score_std",
    "multimodal_score",
    "temporal_score",
)


def feature_matrix(candidates: pd.DataFrame) -> np.ndarray:
    """Build a finite numeric feature matrix from pipeline evidence."""
    missing = [c for c in FEATURE_COLUMNS if c not in candidates.columns]
    if missing:
        raise ValueError(f"missing reranker features: {missing}")
    x = candidates.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


@dataclass
class PairwiseLinearRanker:
    """Small dependency-free RankNet-style linear reranker.

    This is a training primitive, not a claim that fixed weights are optimal.
    It can be trained from official relevance labels once the evaluator/ground
    truth format is available, then serialized and used as a drop-in scorer.
    """

    learning_rate: float = 0.05
    epochs: int = 100
    l2: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, candidates: pd.DataFrame, relevance: np.ndarray) -> "PairwiseLinearRanker":
        x = feature_matrix(candidates)
        y = np.asarray(relevance, dtype=np.float32).reshape(-1)
        if len(x) != len(y) or len(y) < 2:
            raise ValueError("candidates and relevance must have the same length and at least two rows")
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.std_
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.01, size=z.shape[1]).astype(np.float32)
        b = 0.0
        pairs = [(i, j) for i in range(len(y)) for j in range(len(y)) if y[i] > y[j]]
        if not pairs:
            raise ValueError("relevance must contain at least one positive-vs-negative pair")
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for i, j in pairs:
                delta = float(np.dot(w, z[i] - z[j]))
                # sigmoid(-delta) is the derivative of -log(sigmoid(delta)).
                grad = 1.0 / (1.0 + np.exp(np.clip(delta, -40.0, 40.0)))
                diff = z[i] - z[j]
                w += self.learning_rate * (grad * diff - self.l2 * w)
        self.weights_ = w
        self.bias_ = b
        return self

    def score(self, candidates: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("ranker has not been fitted")
        x = feature_matrix(candidates)
        z = (x - self.mean_) / self.std_
        return (z @ self.weights_ + self.bias_).astype(np.float32)

    def rerank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        out = candidates.copy()
        out["learned_rank_score"] = self.score(out)
        return out.sort_values("learned_rank_score", ascending=False).reset_index(drop=True)
