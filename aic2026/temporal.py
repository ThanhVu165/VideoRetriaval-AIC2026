from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalWindow:
    video_id: str
    start_frame: int
    end_frame: int
    score: float

    @property
    def center_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2


def merge_frame_hits(
    video_id: str,
    frame_ids: Iterable[int],
    scores: Iterable[float],
    max_gap: int = 1,
) -> list[TemporalWindow]:
    """Group sparse frame evidence into candidate temporal windows.

    This is deliberately model-agnostic: frame scores may come from CLIP,
    objects, a learned reranker, or a future temporal model.
    """
    pairs = sorted((int(f), float(s)) for f, s in zip(frame_ids, scores))
    if not pairs:
        return []
    windows: list[list[tuple[int, float]]] = [[pairs[0]]]
    for pair in pairs[1:]:
        if pair[0] - windows[-1][-1][0] <= max_gap:
            windows[-1].append(pair)
        else:
            windows.append([pair])

    out: list[TemporalWindow] = []
    for group in windows:
        frames = [x[0] for x in group]
        values = [x[1] for x in group]
        out.append(
            TemporalWindow(
                video_id=video_id,
                start_frame=min(frames),
                end_frame=max(frames),
                score=float(np.max(values)),
            )
        )
    return sorted(out, key=lambda x: x.score, reverse=True)


def refine_window(
    start_frame: int,
    end_frame: int,
    frame_scores: dict[int, float],
) -> int:
    """Select the best observed original frame inside a temporal window."""
    candidates = [(f, s) for f, s in frame_scores.items() if start_frame <= f <= end_frame]
    if not candidates:
        raise ValueError("No scored frames fall inside the requested window")
    return max(candidates, key=lambda item: item[1])[0]
