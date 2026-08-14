from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aic2026.retrieval import FrameIndex


class RetrievalPersistenceContractTest(unittest.TestCase):
    def test_manifest_embedding_contract(self) -> None:
        manifest = pd.DataFrame(
            {
                "video_id": ["V1", "V2"],
                "keyframe_idx": [0, 1],
                "original_frame_id": [10, 20],
                "pts_time": [0.5, 1.0],
                "image_path": ["V1/0.jpg", "V2/1.jpg"],
            }
        )
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = FrameIndex(manifest, embeddings)
        result = index.search_videos(np.asarray([1.0, 0.0], dtype=np.float32), top_k_frames=2, top_k_videos=2)
        self.assertEqual(result.iloc[0]["video_id"], "V1")
        self.assertEqual(int(result.iloc[0]["best_frame_id"]), 10)


if __name__ == "__main__":
    unittest.main()
