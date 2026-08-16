from __future__ import annotations

from aic2026.benchmark import evaluate_results


def test_kis_benchmark_uses_interval_r_score_and_final_score() -> None:
    results = {
        "tkis-01": [
            {"video_id": "V_WRONG", "best_frame_id": 100, "rank_score": 0.99},
            {"video_id": "V1", "best_frame_id": 120, "rank_score": 0.90},
            {"video_id": "V1", "best_frame_id": 150, "rank_score": 0.80},
        ]
    }
    gt = {
        "tkis-01": {
            "task_type": "TKIS",
            "video_ids": ["V1"],
            "intervals": {"V1": [[140, 160]]},
        }
    }

    summary, per_query = evaluate_results(results, gt)

    assert per_query[0]["competition_r@1"] == 0.0
    assert per_query[0]["competition_r@5"] == 1.0
    assert per_query[0]["competition_r@20"] == 1.0
    assert per_query[0]["competition_final_score"] == 0.8
    assert summary["competition_final_score"] == 0.8


def test_local_video_recall_is_not_named_official_competition_score() -> None:
    results = {
        "tkis-02": [
            {"video_id": "V1", "best_frame_id": 10, "rank_score": 0.9},
        ]
    }
    gt = {
        "tkis-02": {
            "task_type": "TKIS",
            "video_ids": ["V1"],
            "intervals": {"V1": [[20, 30]]},
        }
    }

    _, per_query = evaluate_results(results, gt)
    assert per_query[0]["video_recall@1"] == 1.0
    assert per_query[0]["competition_r@1"] == 0.0
