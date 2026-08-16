from __future__ import annotations

import pytest

from aic2026.temporal import merge_frame_hits, refine_window


def test_merge_frame_hits_groups_adjacent_frames() -> None:
    windows = merge_frame_hits("L01_V001", [10, 11, 12, 20], [0.2, 0.9, 0.5, 0.7])
    assert len(windows) == 2
    assert windows[0].start_frame == 10
    assert windows[0].end_frame == 12
    assert windows[0].score == pytest.approx(0.9)


def test_refine_window_returns_best_original_frame() -> None:
    assert refine_window(10, 20, {9: 1.0, 11: 0.2, 15: 0.8, 21: 2.0}) == 15


def test_refine_window_requires_evidence() -> None:
    with pytest.raises(ValueError):
        refine_window(10, 20, {1: 0.2})
