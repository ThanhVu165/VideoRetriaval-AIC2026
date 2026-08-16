from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manual_validation import ValidationRecord, binary_r_at_k


KS = (1, 5, 20, 50, 100)


def load_records(path: str | Path) -> list[ValidationRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Validation file must contain a JSON list")
    return [ValidationRecord(**item) for item in payload]


def build_report(records: list[ValidationRecord]) -> dict[str, Any]:
    by_query: dict[str, list[ValidationRecord]] = {}
    for record in records:
        by_query.setdefault(record.query_id, []).append(record)

    query_reports: list[dict[str, Any]] = []
    for query_id, items in sorted(by_query.items()):
        items = sorted(items, key=lambda x: x.rank)
        task_type = items[0].task_type
        report: dict[str, Any] = {
            "query_id": query_id,
            "task_type": task_type,
            "validated_candidates": len(items),
            "best_valid_rank": None,
        }
        valid_ranks: list[int] = []
        for item in items:
            if task_type.upper() in {"TKIS", "KIS"}:
                valid = item.video_match and item.frame_match
            elif task_type.upper() in {"QA", "Q&A"}:
                valid = item.video_match and item.frame_match and bool(item.answer_match)
            else:
                valid = item.video_match and item.frame_match
            if valid:
                valid_ranks.append(item.rank)
        if valid_ranks:
            report["best_valid_rank"] = min(valid_ranks)
        for k in KS:
            report[f"local_r@{k}"] = binary_r_at_k(items, k)
        query_reports.append(report)

    summary: dict[str, Any] = {
        "queries": len(query_reports),
        "validation_records": len(records),
    }
    if query_reports:
        for k in KS:
            summary[f"mean_local_r@{k}"] = sum(r[f"local_r@{k}"] for r in query_reports) / len(query_reports)
    else:
        for k in KS:
            summary[f"mean_local_r@{k}"] = None
    summary["queries_with_top1"] = sum(r["best_valid_rank"] == 1 for r in query_reports)
    summary["queries_with_any_validated"] = sum(r["best_valid_rank"] is not None for r in query_reports)
    return {"summary": summary, "queries": query_reports}


def write_report(records_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = build_report(load_records(records_path))
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
