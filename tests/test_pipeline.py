from __future__ import annotations

import unittest

import pandas as pd

from aic2026.pipeline import AICPipeline


class PipelineTest(unittest.TestCase):
    def test_multi_frame_aggregation_keeps_best_frame(self) -> None:
        rows = pd.DataFrame(
            {
                "video_id": ["V1", "V1", "V1", "V2"],
                "keyframe_idx": [3, 4, 5, 1],
                "original_frame_id": [30, 40, 50, 10],
                "pts_time": [1.0, 1.3, 1.6, 0.3],
                "score": [0.95, 0.90, 0.85, 0.94],
                "object_path": ["o3", "o4", "o5", "o1"],
            }
        )
        out = AICPipeline._aggregate_frame_evidence(rows, per_video_k=2)
        v1 = out[out.video_id == "V1"].iloc[0]
        self.assertEqual(int(v1.best_frame_id), 30)
        self.assertAlmostEqual(float(v1.retrieval_topk_mean), 0.925, places=5)
        self.assertEqual(str(v1.object_path), "o3")
        self.assertGreater(float(v1.retrieval_score), 0.93)


if __name__ == "__main__":
    unittest.main()
