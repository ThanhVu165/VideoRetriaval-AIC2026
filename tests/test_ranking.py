from __future__ import annotations

import unittest

import pandas as pd

from aic2026.multimodal import lexical_overlap
from aic2026.ranking import rerank_candidates, top_k_submission


class RankingTest(unittest.TestCase):
    def test_lexical_overlap(self) -> None:
        self.assertAlmostEqual(lexical_overlap("red car", "A red car appears"), 1.0)

    def test_rerank_and_topk(self) -> None:
        candidates = pd.DataFrame(
            {
                "video_id": ["V1", "V2"],
                "retrieval_score": [0.9, 0.8],
                "temporal_score": [0.1, 0.9],
                "multimodal_score": [0.1, 0.9],
            }
        )
        ranked = rerank_candidates(candidates)
        self.assertEqual(ranked.iloc[0].video_id, "V2")
        self.assertEqual(len(top_k_submission(ranked, 1)), 1)


if __name__ == "__main__":
    unittest.main()
