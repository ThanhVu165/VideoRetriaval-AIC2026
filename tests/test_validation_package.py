from pathlib import Path

import pandas as pd

from aic2026.query_manifest import QueryRecord
from aic2026.validation_package import candidates_to_validation_rows, write_validation_package


def test_candidate_package_does_not_infer_ground_truth(tmp_path: Path):
    query = QueryRecord(
        query_id="tkis-query-01",
        task_type="TKIS",
        description_vi="mô tả",
        description_en="description",
    )
    candidates = pd.DataFrame(
        [
            {
                "video_id": "v1",
                "semantic_keyframe": 120,
                "temporal_start_frame": 110,
                "temporal_end_frame": 130,
                "retrieval_score": 0.9,
                "multimodal_score": 0.8,
                "temporal_score": 0.7,
                "rank_score": 0.85,
                "best_pts_time": 4.0,
            }
        ]
    )

    rows = candidates_to_validation_rows(query, candidates, top_k=100)
    assert rows[0]["video_id"] == "v1"
    assert "video_match" not in rows[0]
    assert "frame_match" not in rows[0]

    output = write_validation_package(query, candidates, tmp_path, top_k=100)
    payload = output.read_text(encoding="utf-8")
    assert '"official_ground_truth_available": false' in payload
