from __future__ import annotations

import unittest

import numpy as np

from aic2026.alignment import monotonic_event_alignment, select_semantic_keyframe


class AlignmentTest(unittest.TestCase):
    def test_select_keyframe(self) -> None:
        result = select_semantic_keyframe("e1", [10, 11, 12], [0.1, 0.9, 0.2])
        self.assertEqual(result.frame_id, 11)

    def test_monotonic_alignment(self) -> None:
        scores = np.array(
            [
                [0.9, 0.2, 0.1, 0.0],
                [0.0, 0.3, 0.8, 0.2],
                [0.0, 0.1, 0.2, 0.95],
            ],
            dtype=np.float32,
        )
        result = monotonic_event_alignment(["a", "b", "c"], [10, 11, 12, 13], scores, min_separation=1)
        self.assertEqual([x.frame_id for x in result], [10, 12, 13])


if __name__ == "__main__":
    unittest.main()
