from __future__ import annotations

import unittest

from aic2026.competition_metrics import (
    final_score,
    kis_r_score,
    qa_r_score,
    ranking_profile,
    r_at_k,
    trake_r_score,
)


class CompetitionMetricsTest(unittest.TestCase):
    def test_kis_r_score(self) -> None:
        self.assertEqual(kis_r_score("L01_V001", 505, "L01_V001", 500, 510), 1.0)
        self.assertEqual(kis_r_score("L01_V001", 600, "L01_V001", 500, 510), 0.0)
        self.assertEqual(kis_r_score("L02_V003", 505, "L01_V001", 500, 510), 0.0)

    def test_qa_requires_video_frame_and_semantic_match(self) -> None:
        exact = lambda pred, gt: pred.strip().casefold() == gt.strip().casefold()
        self.assertEqual(
            qa_r_score("L05_V005", 888, "màu xanh", "L05_V005", 800, 900, "màu xanh", exact),
            1.0,
        )
        self.assertEqual(
            qa_r_score("L05_V005", 888, "màu trắng", "L05_V005", 800, 900, "màu xanh", exact),
            0.0,
        )

    def test_trake_wrong_video_is_zero(self) -> None:
        intervals = [(95, 105), (145, 155), (195, 205), (245, 255)]
        self.assertEqual(trake_r_score("WRONG", [101, 146, 203, 251], "L10_V010", intervals), 0.0)

    def test_trake_fraction_when_video_is_correct(self) -> None:
        intervals = [(95, 105), (145, 155), (195, 205), (245, 255)]
        self.assertAlmostEqual(
            trake_r_score("L10_V010", [101, 156, 203, 251], "L10_V010", intervals),
            0.75,
        )

    def test_final_score_uses_maximum_within_each_cutoff(self) -> None:
        scores = [0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]
        self.assertEqual(r_at_k(scores, 1), 0.5)
        self.assertEqual(r_at_k(scores, 5), 0.8)
        self.assertEqual(r_at_k(scores, 20), 0.8)
        self.assertAlmostEqual(final_score(scores), 0.74)
        self.assertEqual(ranking_profile(scores)["r@100"], 0.8)


if __name__ == "__main__":
    unittest.main()
