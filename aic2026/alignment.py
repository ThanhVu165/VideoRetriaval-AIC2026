from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SemanticKeyframe:
    event_id: str
    frame_id: int
    score: float
    rank: int


def select_semantic_keyframe(
    event_id: str,
    frame_ids: Sequence[int],
    scores: Sequence[float],
    min_frame: int | None = None,
    max_frame: int | None = None,
) -> SemanticKeyframe:
    """Select the semantic keyframe with the highest event-frame score."""
    if len(frame_ids) != len(scores) or not frame_ids:
        raise ValueError("frame_ids and scores must be non-empty and have equal length")
    candidates = [
        (int(frame), float(score), idx)
        for idx, (frame, score) in enumerate(zip(frame_ids, scores))
        if (min_frame is None or int(frame) >= min_frame)
        and (max_frame is None or int(frame) <= max_frame)
    ]
    if not candidates:
        raise ValueError("No candidate frames inside requested interval")
    candidates.sort(key=lambda x: x[1], reverse=True)
    frame, score, rank = candidates[0]
    return SemanticKeyframe(event_id=event_id, frame_id=frame, score=score, rank=rank)


def monotonic_event_alignment(
    event_ids: Sequence[str],
    frame_ids: Sequence[int],
    score_matrix: np.ndarray,
    min_separation: int = 0,
) -> list[SemanticKeyframe]:
    """Align an ordered event sequence to monotonically increasing frames.

    score_matrix[e, f] is the compatibility between event e and frame f.
    Dynamic programming prevents a later event from being assigned to an
    earlier frame and is suitable for TRAKE-style ordered event sequences.
    """
    scores = np.asarray(score_matrix, dtype=np.float32)
    n_events, n_frames = scores.shape
    if n_events != len(event_ids) or n_frames != len(frame_ids):
        raise ValueError("score_matrix shape does not match event/frame inputs")
    if n_events == 0 or n_frames == 0:
        return []
    if min_separation < 0:
        raise ValueError("min_separation must be non-negative")

    neg_inf = -np.inf
    dp = np.full((n_events, n_frames), neg_inf, dtype=np.float32)
    back = np.full((n_events, n_frames), -1, dtype=np.int32)
    dp[0] = scores[0]

    for e in range(1, n_events):
        best_value = neg_inf
        best_index = -1
        for f in range(n_frames):
            prev_limit = f - min_separation
            if prev_limit >= 0:
                value = dp[e - 1, prev_limit]
                if value > best_value:
                    best_value = value
                    best_index = prev_limit
            if best_index >= 0:
                dp[e, f] = scores[e, f] + best_value
                back[e, f] = best_index

    last = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, last]):
        raise ValueError("Unable to find a monotonic alignment for all events")

    selected = [last]
    for e in range(n_events - 1, 0, -1):
        selected.append(int(back[e, selected[-1]]))
    selected.reverse()

    return [
        SemanticKeyframe(
            event_id=str(event_ids[e]),
            frame_id=int(frame_ids[f]),
            score=float(scores[e, f]),
            rank=f,
        )
        for e, f in enumerate(selected)
    ]
