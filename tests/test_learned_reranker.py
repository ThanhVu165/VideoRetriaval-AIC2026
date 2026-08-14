from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aic2026.learned_reranker import PairwiseLinearRanker


class LearnedRerankerTest(unittest.TestCase):
    def test_pairwise_training_orders_relevant_candidate(self) -> None:
        candidates = pd.DataFrame(
            {
                "retrieval_score": [0.9, 0.4, 0.2],
                "retrieval_best_score": [0.9, 0.4, 0.2],
                "retrieval_topk_mean": [0.85, 0.35, 0.15],
                "retrieval_score_std": [0.02, 0.04, 0.03],
                "multimodal_score": [0.8, 0.3, 0.1],
                "temporal_score": [0.9, 0.2, 0.1],
            }
        )
        ranker = PairwiseLinearRanker(learning_rate=0.05, epochs=20).fit(candidates, np.array([2, 1, 0]))
        ranked = ranker.rerank(candidates)
        self.assertEqual(int(ranked.index[0]), 0)


if __name__ == "__main__":
    unittest.main()
