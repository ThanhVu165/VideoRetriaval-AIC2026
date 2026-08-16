import numpy as np

from aic2026.alignment import monotonic_event_alignment


def test_trake_alignment_preserves_event_order():
    event_ids = ["event_1", "event_2", "event_3"]
    frame_ids = [10, 11, 12, 13, 14]
    scores = np.asarray(
        [
            [0.9, 0.8, 0.1, 0.0, 0.0],
            [0.0, 0.2, 0.9, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.3, 0.95],
        ],
        dtype=np.float32,
    )
    result = monotonic_event_alignment(event_ids, frame_ids, scores)
    assert [item.event_id for item in result] == event_ids
    assert [item.frame_id for item in result] == [10, 12, 14]
