from __future__ import annotations

import json
from pathlib import Path

from .retrieval import FrameIndex


def build_faiss_index(
    manifest: str | Path,
    embeddings: str | Path,
    output_index: str | Path,
    metadata_output: str | Path | None = None,
) -> dict[str, object]:
    """Build a persistent exact-cosine FAISS frame index.

    The generated index contains only vectors. Row metadata stays in the
    unified manifest, preventing duplicated metadata and preserving the
    canonical mapping between keyframe_idx and original_frame_id.
    """
    index = FrameIndex.from_files(manifest, embeddings, backend="numpy")
    output = Path(output_index)
    index.persist_faiss(output)

    report: dict[str, object] = {
        "index": str(output),
        "manifest": str(manifest),
        "embeddings": str(embeddings),
        "rows": len(index.manifest),
        "dimension": int(index.embeddings.shape[1]),
        "metric": "inner_product_on_l2_normalized_embeddings",
        "backend": "faiss.IndexFlatIP",
    }
    if metadata_output is not None:
        meta_path = Path(metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
