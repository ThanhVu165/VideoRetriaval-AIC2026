from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Backend = Literal["numpy", "faiss"]
VideoAggregation = Literal["max", "topk_mean"]


@dataclass(frozen=True)
class SearchResult:
    video_id: str
    keyframe_idx: int
    original_frame_id: int
    pts_time: float
    score: float
    image_path: str
    object_path: str = ""


class FrameIndex:
    """Frame-level CLIP retrieval index.

    The index consumes the unified manifest produced by Phase 0 and a matrix
    of normalized frame embeddings. It deliberately accepts a precomputed
    query embedding: the exact BTC query/CLIP text-encoding protocol is not
    assumed until the official query package is available.
    """

    def __init__(self, manifest: pd.DataFrame, embeddings: np.ndarray, backend: Backend = "numpy"):
        if len(manifest) != len(embeddings):
            raise ValueError(f"manifest rows ({len(manifest)}) != embeddings ({len(embeddings)})")
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {embeddings.ndim}-D")
        required = {"video_id", "keyframe_idx", "original_frame_id", "pts_time", "image_path"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")

        self.manifest = manifest.reset_index(drop=True).copy()
        x = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero-norm frame embedding found")
        self.embeddings = x / norms
        self.backend = backend
        self._faiss = None

        if backend == "faiss":
            try:
                import faiss  # type: ignore
            except ImportError as exc:
                raise ImportError("FAISS backend requested but faiss is not installed") from exc
            self._faiss = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._faiss.add(self.embeddings)

    @classmethod
    def from_files(cls, manifest_path: str | Path, embedding_path: str | Path, backend: Backend = "numpy") -> "FrameIndex":
        manifest = pd.read_parquet(manifest_path) if str(manifest_path).endswith(".parquet") else pd.read_csv(manifest_path)
        embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        return cls(manifest, embeddings, backend=backend)

    def search_frames(self, query_embedding: np.ndarray, top_k: int = 100) -> list[SearchResult]:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(f"query dimension {q.shape[0]} != index dimension {self.embeddings.shape[1]}")
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("zero-norm query embedding")
        q /= norm
        top_k = max(1, min(int(top_k), len(self.manifest)))

        if self._faiss is not None:
            scores, indices = self._faiss.search(q[None, :], top_k)
            indices = indices[0]
            scores = scores[0]
        else:
            scores_all = self.embeddings @ q
            indices = np.argpartition(-scores_all, top_k - 1)[:top_k]
            indices = indices[np.argsort(-scores_all[indices])]
            scores = scores_all[indices]

        out: list[SearchResult] = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            row = self.manifest.iloc[int(idx)]
            out.append(
                SearchResult(
                    video_id=str(row.video_id),
                    keyframe_idx=int(row.keyframe_idx),
                    original_frame_id=int(row.original_frame_id),
                    pts_time=float(row.pts_time),
                    score=float(score),
                    image_path=str(row.image_path),
                    object_path=str(row.get("object_path", "")),
                )
            )
        return out

    def search_videos(
        self,
        query_embedding: np.ndarray,
        top_k_frames: int = 200,
        top_k_videos: int = 100,
        aggregation: VideoAggregation = "max",
        per_video_k: int = 3,
    ) -> pd.DataFrame:
        """Return ranked candidate videos from frame retrieval."""
        frames = self.search_frames(query_embedding, top_k=top_k_frames)
        if not frames:
            return pd.DataFrame(columns=["video_id", "score", "best_frame_idx", "best_frame_id", "best_pts_time"])

        rows = pd.DataFrame([r.__dict__ for r in frames])
        if aggregation == "max":
            grouped = rows.groupby("video_id", sort=False)
            result = grouped["score"].max().rename("score").reset_index()
        elif aggregation == "topk_mean":
            result = (
                rows.sort_values(["video_id", "score"], ascending=[True, False])
                .groupby("video_id", sort=False)["score"]
                .apply(lambda s: s.head(per_video_k).mean())
                .rename("score")
                .reset_index()
            )
        else:
            raise ValueError(f"unsupported aggregation: {aggregation}")

        best = rows.sort_values("score", ascending=False).drop_duplicates("video_id")
        result = result.merge(
            best[["video_id", "keyframe_idx", "original_frame_id", "pts_time", "object_path"]],
            on="video_id",
            how="left",
        ).rename(columns={
            "keyframe_idx": "best_frame_idx",
            "original_frame_id": "best_frame_id",
            "pts_time": "best_pts_time",
        })
        return result.sort_values("score", ascending=False).head(top_k_videos).reset_index(drop=True)
