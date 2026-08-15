from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Sequence

import pandas as pd

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(str(text)) if len(x) > 1}


def lexical_overlap(query: str, text: str) -> float:
    q = tokenize(query)
    d = tokenize(text)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def _pick(columns: Sequence[str], names: Iterable[str]) -> str | None:
    normalized = {c.lower().strip(): c for c in columns}
    for name in names:
        if name in normalized:
            return normalized[name]
    for c in columns:
        lc = c.lower().strip()
        if any(name in lc for name in names):
            return c
    return None


@dataclass(frozen=True)
class EvidenceConfig:
    """Schema hints for an evidence source.

    Empty hints are resolved automatically from the SQLite table schema.
    """

    table: str | None = None
    video_column: str | None = None
    frame_column: str | None = None
    text_column: str | None = None
    time_column: str | None = None


class EvidenceStore:
    name: str

    def score_candidates(self, query: str, rows: pd.DataFrame) -> list[float]:
        raise NotImplementedError


class SQLiteEvidenceStore(EvidenceStore):
    """Read ASR/OCR/caption-like evidence without assuming one fixed schema.

    The adapter inspects tables once, then fetches evidence only for the
    retrieved candidates. This keeps auxiliary modalities optional and avoids
    scanning the entire artifact for every query.
    """

    def __init__(self, path: str | Path, name: str, config: EvidenceConfig | None = None):
        self.path = Path(path)
        self.name = name
        self.config = config or EvidenceConfig()
        self.table: str | None = None
        self.video_column: str | None = None
        self.frame_column: str | None = None
        self.text_column: str | None = None
        self.time_column: str | None = None
        self._ready = False
        self._prepare()

    @property
    def available(self) -> bool:
        return self._ready

    def _prepare(self) -> None:
        if not self.path.exists():
            return
        try:
            with sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True) as con:
                tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                if not tables:
                    return
                table = self.config.table
                if table not in tables:
                    table = None
                for candidate in ([table] if table else tables):
                    if not candidate:
                        continue
                    info = con.execute(f'PRAGMA table_info("{candidate}")').fetchall()
                    columns = [str(r[1]) for r in info]
                    video = self.config.video_column or _pick(
                        columns, ("video_id", "video", "vid", "videoid")
                    )
                    frame = self.config.frame_column or _pick(
                        columns, ("frame_id", "frame_idx", "frame", "keyframe_id", "original_frame_id")
                    )
                    text = self.config.text_column or _pick(
                        columns, ("text", "transcript", "asr", "ocr", "caption", "content", "value", "description")
                    )
                    time_col = self.config.time_column or _pick(
                        columns, ("pts_time", "timestamp", "time", "start_time")
                    )
                    if video and text:
                        self.table = candidate
                        self.video_column = video
                        self.frame_column = frame
                        self.text_column = text
                        self.time_column = time_col
                        self._ready = True
                        return
        except (OSError, sqlite3.Error):
            return

    @staticmethod
    def _norm_id(value: object) -> str:
        return str(value)

    def score_candidates(self, query: str, rows: pd.DataFrame) -> list[float]:
        if rows.empty or not self._ready:
            return [0.0] * len(rows)
        assert self.table and self.video_column and self.text_column
        scores: list[float] = []
        try:
            with sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True) as con:
                qcols = f'"{self.video_column}", "{self.text_column}"'
                if self.frame_column:
                    qcols += f', "{self.frame_column}"'
                if self.time_column:
                    qcols += f', "{self.time_column}"'
                sql = f'SELECT {qcols} FROM "{self.table}" WHERE "{self.video_column}" = ?'
                cache: dict[str, list[tuple[str, object, object | None, object | None]]] = {}
                for _, row in rows.iterrows():
                    video_id = self._norm_id(row["video_id"])
                    if video_id not in cache:
                        try:
                            raw = con.execute(sql, (video_id,)).fetchall()
                        except sqlite3.Error:
                            raw = []
                        cache[video_id] = [
                            (str(r[1] or ""), r[2] if self.frame_column else None, r[3] if self.time_column else None)
                            for r in raw
                        ]
                    evidence = cache[video_id]
                    target_frame = row.get("best_frame_id", row.get("original_frame_id", None))
                    best = 0.0
                    for text, frame_id, timestamp in evidence:
                        score = lexical_overlap(query, text)
                        if frame_id is not None and target_frame is not None:
                            try:
                                distance = abs(int(frame_id) - int(target_frame))
                                if distance <= 30:
                                    score *= 1.0
                                else:
                                    score *= 0.5
                            except (TypeError, ValueError):
                                pass
                        best = max(best, score)
                    scores.append(best)
        except (OSError, sqlite3.Error):
            return [0.0] * len(rows)
        return scores


class JsonEvidenceStore(EvidenceStore):
    """Optional JSON/JSONL evidence adapter for development artifacts."""

    def __init__(self, path: str | Path, name: str):
        self.path = Path(path)
        self.name = name
        self._rows: list[dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            if self.path.suffix.lower() == ".jsonl":
                with self.path.open("r", encoding="utf-8") as f:
                    self._rows = [json.loads(line) for line in f if line.strip()]
            else:
                obj = json.loads(self.path.read_text(encoding="utf-8"))
                self._rows = obj if isinstance(obj, list) else [obj]
        except (OSError, ValueError, TypeError):
            self._rows = []

    def score_candidates(self, query: str, rows: pd.DataFrame) -> list[float]:
        if rows.empty or not self._rows:
            return [0.0] * len(rows)
        out: list[float] = []
        for _, row in rows.iterrows():
            vid = str(row["video_id"])
            frame = row.get("best_frame_id", None)
            best = 0.0
            for item in self._rows:
                if str(item.get("video_id", item.get("video", ""))) != vid:
                    continue
                text = item.get("text", item.get("transcript", item.get("caption", item.get("ocr", ""))))
                score = lexical_overlap(query, str(text))
                item_frame = item.get("frame_id", item.get("frame_idx"))
                if frame is not None and item_frame is not None:
                    try:
                        if abs(int(item_frame) - int(frame)) > 30:
                            score *= 0.5
                    except (TypeError, ValueError):
                        pass
                best = max(best, score)
            out.append(best)
        return out
