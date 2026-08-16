from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Any

import pandas as pd


_EVENT_RE = re.compile(r"(?m)^\s*(E\d+)\s*:\s*(.+?)(?=\n\s*E\d+\s*:|\Z)", re.DOTALL)


@dataclass(frozen=True)
class QueryRecord:
    """Canonical local representation of a row from the supplied query workbook.

    This is a repository input contract, not an organizer submission schema.
    """

    query_id: str
    task_type: str
    description_vi: str
    description_en: str
    events: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_task_type(query_id: str) -> str:
    """Classify the supplied workbook's observed query-id convention."""
    value = str(query_id).strip().lower()
    if value.startswith("tkis-query-"):
        return "TKIS"
    if value.startswith("qa-query-"):
        return "QA"
    if value.startswith("trake-"):
        return "TRAKE"
    if value.startswith("vkis-"):
        return "VKIS"
    return "UNKNOWN"


def extract_events(description: str) -> tuple[dict[str, str], ...]:
    """Extract explicit E1/E2/... blocks from a TRAKE description."""
    events: list[dict[str, str]] = []
    for event_id, text in _EVENT_RE.findall(str(description or "")):
        events.append({"event_id": event_id, "text": " ".join(text.split())})
    return tuple(events)


def load_query_manifest(path: str | Path) -> list[QueryRecord]:
    """Load the supplied AIC query workbook without inventing GT/submission fields."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix == ".json":
        payload = pd.read_json(source)
        frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    else:
        raise ValueError(f"Unsupported query source: {source.suffix}")

    required = {"Query Name", "Description", "Trans"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Query source is missing required columns: {sorted(missing)}")

    records: list[QueryRecord] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        query_id = str(row["Query Name"]).strip()
        if not query_id or query_id.lower() == "nan":
            raise ValueError("Query source contains an empty Query Name")
        if query_id in seen:
            raise ValueError(f"Duplicate Query Name: {query_id}")
        seen.add(query_id)

        description_vi = str(row["Description"] if pd.notna(row["Description"]) else "").strip()
        description_en = str(row["Trans"] if pd.notna(row["Trans"]) else "").strip()
        task_type = infer_task_type(query_id)
        events = extract_events(description_vi) if task_type == "TRAKE" else ()
        records.append(
            QueryRecord(
                query_id=query_id,
                task_type=task_type,
                description_vi=description_vi,
                description_en=description_en,
                events=events,
            )
        )
    return records


def manifest_report(records: list[QueryRecord]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    for record in records:
        counts[record.task_type] = counts.get(record.task_type, 0) + 1
        if record.task_type == "TRAKE":
            event_counts[record.query_id] = len(record.events)
    return {
        "queries": len(records),
        "task_counts": counts,
        "trake_event_counts": event_counts,
        "query_ids": [record.query_id for record in records],
    }
