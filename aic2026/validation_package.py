from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import pandas as pd

from .query_manifest import QueryRecord


OUTPUT_COLUMNS = [
    "query_id",
    "task_type",
    "rank",
    "video_id",
    "semantic_keyframe",
    "temporal_start_frame",
    "temporal_end_frame",
    "retrieval_score",
    "multimodal_score",
    "temporal_score",
    "final_score",
    "best_pts_time",
]


def candidates_to_validation_rows(
    query: QueryRecord,
    candidates: pd.DataFrame,
    top_k: int = 100,
) -> list[dict[str, Any]]:
    """Convert one ranked pipeline result into human-reviewable records.

    This function deliberately stores model outputs only. It does not infer
    whether a candidate is correct.
    """
    if candidates.empty:
        return []
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(candidates.head(top_k).iterrows(), start=1):
        item: dict[str, Any] = {
            "query_id": query.query_id,
            "task_type": query.task_type,
            "rank": rank,
        }
        for column in OUTPUT_COLUMNS[3:]:
            value = row[column] if column in row.index else None
            if pd.isna(value) if value is not None else False:
                value = None
            if hasattr(value, "item"):
                value = value.item()
            item[column] = value
        rows.append(item)
    return rows


def write_validation_package(
    query: QueryRecord,
    candidates: pd.DataFrame,
    output_dir: str | Path,
    top_k: int = 100,
) -> Path:
    """Write one query's ranked candidates as JSON for manual inspection."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{query.query_id}.json"
    payload = {
        "query": query.to_dict(),
        "source": "model_ranked_candidates",
        "official_ground_truth_available": False,
        "candidates": candidates_to_validation_rows(query, candidates, top_k=top_k),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
