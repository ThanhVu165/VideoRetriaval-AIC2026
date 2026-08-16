from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .evidence import EvidenceStore, lexical_overlap

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


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


def minmax_normalize(values: Iterable[float]) -> list[float]:
    xs = [float(x) for x in values]
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo <= 1e-12:
        return [1.0 if hi > 0 else 0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _rank_fusion(scores: Sequence[float], rrf_k: int) -> list[float]:
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    fused = [0.0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        if scores[idx] > 0:
            fused[idx] = 1.0 / (rrf_k + rank)
    return fused


@dataclass(frozen=True)
class FusionWeights:
    clip: float = 0.70
    objects: float = 0.10
    metadata: float = 0.05
    evidence: float = 0.15


class MultimodalReranker:
    """Candidate reranker with optional rank-level evidence fusion.

    CLIP remains the primary signal. Auxiliary object/metadata overlap is kept
    for compatibility, while ASR/OCR/caption-like stores are fused by RRF so
    their native score scales never need to be calibrated against CLIP.
    """

    def __init__(
        self,
        weights: FusionWeights | None = None,
        evidence_stores: Sequence[EvidenceStore] | None = None,
        rrf_k: int = 60,
    ):
        self.weights = weights or FusionWeights()
        self.evidence_stores = tuple(evidence_stores or ())
        self.rrf_k = max(1, int(rrf_k))

    def score_manifest(self, query: str, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        out = rows.copy()
        clip = minmax_normalize(out["score"].tolist())
        object_scores = [load_json_text(p) for p in out.get("object_path", pd.Series([""] * len(out)))]
        object_scores = [lexical_overlap(query, text) for text in object_scores]
        metadata_scores = [
            lexical_overlap(query, text)
            for text in out.get("metadata_text", pd.Series([""] * len(out)))
        ]

        evidence_components: list[list[float]] = []
        evidence_names: list[str] = []
        for store in self.evidence_stores:
            try:
                evidence_components.append(store.score_candidates(query, out))
                evidence_names.append(store.name)
            except (OSError, ValueError, TypeError, KeyError):
                evidence_components.append([0.0] * len(out))
                evidence_names.append(store.name)

        if evidence_components:
            rrfs = [_rank_fusion(scores, self.rrf_k) for scores in evidence_components]
            evidence_raw = [sum(component[i] for component in rrfs) for i in range(len(out))]
            evidence_norm = minmax_normalize(evidence_raw)
        else:
            evidence_norm = [0.0] * len(out)

        out["clip_norm"] = clip
        out["object_score"] = object_scores
        out["metadata_score"] = metadata_scores
        out["evidence_score"] = evidence_norm
        out["evidence_modalities"] = ",".join(evidence_names)
        out["fused_score"] = (
            self.weights.clip * out["clip_norm"]
            + self.weights.objects * out["object_score"]
            + self.weights.metadata * out["metadata_score"]
            + self.weights.evidence * out["evidence_score"]
        )
        return out.sort_values("fused_score", ascending=False).reset_index(drop=True)
