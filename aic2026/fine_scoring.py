from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class FrameScorerProtocol(Protocol):
    def __call__(self, frames: Sequence[np.ndarray]) -> Sequence[float]: ...


@dataclass(frozen=True)
class TemporalScoreConfig:
    """Controls conversion of dense frame scores into a temporal event score."""

    temperature: float = 0.07
    top_fraction: float = 0.20
    continuity_bonus: float = 0.05


def normalize_scores(scores: Sequence[float]) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo <= 1e-8:
        return np.ones_like(x) if hi > 0 else np.zeros_like(x)
    return (x - lo) / (hi - lo)


def temporal_event_score(frame_ids: Sequence[int], scores: Sequence[float], config: TemporalScoreConfig | None = None) -> float:
    cfg = config or TemporalScoreConfig()
    if len(frame_ids) != len(scores) or not scores:
        return 0.0
    x = normalize_scores(scores)
    k = max(1, int(np.ceil(len(x) * cfg.top_fraction)))
    top = np.sort(x)[-k:]
    base = float(top.mean())
    if len(x) < 2:
        return base
    order = np.argsort(x)[::-1]
    best = int(order[0])
    neighbor = max((i for i in (best - 1, best + 1) if 0 <= i < len(x)), key=lambda i: x[i], default=-1)
    continuity = float(x[neighbor]) if neighbor >= 0 else 0.0
    return base + cfg.continuity_bonus * continuity


def select_peak_frame(frame_ids: Sequence[int], scores: Sequence[float]) -> int:
    if len(frame_ids) != len(scores) or not frame_ids:
        raise ValueError("frame_ids and scores must be non-empty and have equal length")
    return int(frame_ids[int(np.argmax(np.asarray(scores, dtype=np.float32)))])
