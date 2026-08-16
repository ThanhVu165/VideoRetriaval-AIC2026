from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ValidationRecord:
    """Human-verified local label; never represents official organizer GT."""

    query_id: str
    task_type: str
    rank: int
    video_id: str
    frame_id: int | None = None
    video_match: bool = False
    frame_match: bool = False
    answer_match: bool | None = None
    event_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_records(records: Iterable[ValidationRecord], path: str | Path) -> None:
    payload = [record.to_dict() for record in records]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_records(path: str | Path) -> list[ValidationRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manual validation file must contain a JSON list")
    return [ValidationRecord(**item) for item in payload]


def first_valid_rank(records: Iterable[ValidationRecord]) -> int | None:
    """Return the best rank with a locally verified complete match."""
    valid: list[int] = []
    for record in records:
        if record.task_type.upper() in {"TKIS", "KIS"}:
            ok = record.video_match and record.frame_match
        elif record.task_type.upper() in {"QA", "Q&A"}:
            ok = record.video_match and record.frame_match and bool(record.answer_match)
        else:
            ok = record.video_match and record.frame_match
        if ok:
            valid.append(int(record.rank))
    return min(valid) if valid else None


def binary_r_at_k(records: Iterable[ValidationRecord], k: int) -> float:
    """Local diagnostic: whether a complete human-verified answer occurs in Top-k."""
    best = first_valid_rank(record for record in records if int(record.rank) <= k)
    return 1.0 if best is not None else 0.0
