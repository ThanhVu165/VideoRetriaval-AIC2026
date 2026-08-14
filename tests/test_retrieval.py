from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aic2026.retrieval import FrameIndex


class FrameIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = pd.DataFrame(
            {
                "video_id": ["V1", "V1", "V2"],
                "keyframe_idx": [0, 1, 0],
                "original_frame_id": [0, 30, 0],
                "pts_time": [0.0, 1.0, 0.0],
                "image_path": ["V1/0000.jpg", "V1/0001.jpg", "V2/0000.jpg"],
            }
        )
        self.embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def test_frame_search(self) -> None:
        index = FrameIndex(self.manifest, self.embeddings)
        result = index.search_frames(np.array([1.0, 0.0]), top_k=2)
        self.assertEqual(result[0].video_id, "V1")
        self.assertEqual(result[0].original_frame_id, 0)
        self.assertGreaterEqual(result[0].score, result[1].score)

    def test_video_search_max(self) -> None:
        index = FrameIndex(self.manifest, self.embeddings)
        result = index.search_videos(np.array([1.0, 0.0]), top_k_frames=3, top_k_videos=2)
        self.assertEqual(result.iloc[0].video_id, "V1")
        self.assertEqual(int(result.iloc[0].best_frame_id), 0)

    def test_dimension_validation(self) -> None:
        index = FrameIndex(self.manifest, self.embeddings)
        with self.assertRaises(ValueError):
            index.search_frames(np.array([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
