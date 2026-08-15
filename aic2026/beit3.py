from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class BEiT3Index:
    """Read-only BEiT-3 FAISS index aligned to the unified frame manifest.

    The reference artifact is an IndexIDMap2 whose IDs are expected to map
    directly to manifest row indices. This class validates that invariant
    before serving search results.
    """

    def __init__(self, manifest: pd.DataFrame, index, name: str = "beit3") -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.index = index
        self.name = name
        self._validate()

    @classmethod
    def from_files(
        cls,
        manifest_path: str | Path,
        index_path: str | Path,
    ) -> "BEiT3Index":
        import faiss

        manifest = (
            pd.read_parquet(manifest_path)
            if str(manifest_path).lower().endswith(".parquet")
            else pd.read_csv(manifest_path)
        )
        index = faiss.read_index(str(index_path))
        return cls(manifest, index)

    def _validate(self) -> None:
        import faiss

        if not hasattr(self.index, "id_map"):
            raise ValueError("BEiT-3 index must be an ID-mapped FAISS index")
        ids = faiss.vector_to_array(self.index.id_map)
        n = len(self.manifest)
        if self.index.ntotal != n:
            raise ValueError(
                f"BEiT-3 ntotal ({self.index.ntotal}) != manifest rows ({n})"
            )
        expected = np.arange(n, dtype=ids.dtype)
        if not np.array_equal(ids, expected):
            raise ValueError(
                "BEiT-3 IDs are not exactly aligned to manifest row indices"
            )
        if self.index.metric_type != 0:
            raise ValueError(
                f"BEiT-3 index metric {self.index.metric_type} is not inner product"
            )

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 100) -> pd.DataFrame:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension {q.shape[1]} != BEiT-3 dimension {self.dimension}"
            )
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("zero-norm BEiT-3 query embedding")
        q = q / norm
        top_k = max(1, min(int(top_k), self.size))
        scores, ids = self.index.search(q, top_k)

        rows: list[dict[str, object]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self.manifest.iloc[int(idx)]
            rows.append(
                {
                    "evidence_id": int(idx),
                    "video_id": str(row["video_id"]),
                    "keyframe_idx": int(row["keyframe_idx"]),
                    "original_frame_id": int(row["original_frame_id"]),
                    "pts_time": float(row["pts_time"]),
                    "score": float(score),
                    "image_path": str(row["image_path"]),
                }
            )
        return pd.DataFrame(rows)

    def diagnostics(self) -> dict[str, object]:
        import faiss

        ids = faiss.vector_to_array(self.index.id_map)
        return {
            "name": self.name,
            "size": self.size,
            "dimension": self.dimension,
            "metric": int(self.index.metric_type),
            "id_min": int(ids.min()),
            "id_max": int(ids.max()),
            "sequential_ids": bool(
                np.array_equal(ids, np.arange(len(ids), dtype=ids.dtype))
            ),
        }
