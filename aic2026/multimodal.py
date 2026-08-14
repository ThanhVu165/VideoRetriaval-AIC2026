from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(str(text)) if len(x) > 1}


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            if key.lower() in {"label", "name", "class", "category", "description", "caption", "title", "text"}:
                if isinstance(item, (str, int, float)):
                    out.append(str(item))
            out.extend(_flatten_text(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_text(item))
        return out
    return []


def load_json_text(path: str | Path) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return " ".join(_flatten_text(obj))
    except (OSError, ValueError, TypeError):
        return ""


def lexical_overlap(query: str, text: str) -> float:
    q = _tokens(query)
    d = _tokens(text)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def minmax_normalize(values: Iterable[float]) -> list[float]:
    xs = [float(x) for x in values]
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo <= 1e-12:
        return [1.0 if hi > 0 else 0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


@dataclass(frozen=True)
class FusionWeights:
    clip: float = 0.70
    objects: float = 0.20
    metadata: float = 0.10


class MultimodalReranker:
    """Lightweight auxiliary reranker for candidate frames/videos.

    CLIP remains the primary semantic signal. Object and metadata overlap are
    intentionally auxiliary and can later be replaced by learned scores
    without changing the pipeline contract.
    """

    def __init__(self, weights: FusionWeights | None = None):
        self.weights = weights or FusionWeights()

    def score_manifest(self, query: str, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        out = rows.copy()
        clip = minmax_normalize(out["score"].tolist())
        object_scores = [load_json_text(p) for p in out.get("object_path", pd.Series([""] * len(out)))]
        object_scores = [lexical_overlap(query, text) for text in object_scores]
        metadata_scores = [lexical_overlap(query, text) for text in out.get("metadata_text", pd.Series([""] * len(out)))]
        out["clip_norm"] = clip
        out["object_score"] = object_scores
        out["metadata_score"] = metadata_scores
        out["fused_score"] = (
            self.weights.clip * out["clip_norm"]
            + self.weights.objects * out["object_score"]
            + self.weights.metadata * out["metadata_score"]
        )
        return out.sort_values("fused_score", ascending=False).reset_index(drop=True)
