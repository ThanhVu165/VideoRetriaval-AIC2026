from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .support_data import resolve_object_path


def _numeric_image_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**18, path.name


def _discover_images(keyframe_video_dir: Path) -> list[Path]:
    images = [
        p for p in keyframe_video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(images, key=_numeric_image_key)


def build_unified_dataset(
    clip_dir: str | Path,
    mapping_dir: str | Path,
    keyframes_dir: str | Path,
    output_manifest: str | Path,
    output_embeddings: str | Path,
    report_output: str | Path | None = None,
    objects_dir: str | Path | None = None,
) -> dict[str, object]:
    """Flatten BTC CLIP/keyframe/mapping data and attach object JSON paths.

    Alignment is validated independently for every video:
    CLIP row i <-> mapping row i <-> numerically sorted keyframe i.
    ``mapping.frame_idx`` is preserved as the original source-video frame ID.
    """
    clip_root = Path(clip_dir)
    mapping_root = Path(mapping_dir)
    keyframe_root = Path(keyframes_dir)
    manifest_path = Path(output_manifest)
    embedding_path = Path(output_embeddings)

    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    errors: list[dict[str, str]] = []
    object_hits = 0

    clip_files = sorted(clip_root.glob("*.npy"))
    for clip_path in clip_files:
        video_id = clip_path.stem
        mapping_path = mapping_root / f"{video_id}.csv"
        video_keyframes = keyframe_root / video_id
        try:
            if not mapping_path.exists():
                raise FileNotFoundError(f"missing mapping: {mapping_path}")
            if not video_keyframes.is_dir():
                raise FileNotFoundError(f"missing keyframes: {video_keyframes}")

            embeddings = np.load(clip_path, allow_pickle=False)
            if embeddings.ndim != 2 or embeddings.shape[1] != 512:
                raise ValueError(f"expected CLIP shape (N, 512), got {embeddings.shape}")

            mapping = pd.read_csv(mapping_path)
            required = {"n", "pts_time", "fps", "frame_idx"}
            missing = required - set(mapping.columns)
            if missing:
                raise ValueError(f"mapping missing columns: {sorted(missing)}")

            images = _discover_images(video_keyframes)
            if not (len(embeddings) == len(mapping) == len(images)):
                raise ValueError(
                    f"alignment mismatch: clip={len(embeddings)}, mapping={len(mapping)}, images={len(images)}"
                )

            for i, image_path in enumerate(images):
                map_row = mapping.iloc[i]
                expected_n = i + 1
                if int(map_row["n"]) != expected_n:
                    raise ValueError(f"mapping n is not contiguous at row {i}: {map_row['n']}")
                object_path = resolve_object_path(objects_dir, video_id, image_path)
                object_hits += bool(object_path)
                rows.append({
                    "video_id": video_id,
                    "keyframe_idx": int(map_row["n"]),
                    "original_frame_id": int(map_row["frame_idx"]),
                    "pts_time": float(map_row["pts_time"]),
                    "fps": float(map_row["fps"]),
                    "image_path": str(image_path),
                    "object_path": object_path,
                })
            vectors.append(np.asarray(embeddings, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            errors.append({"video_id": video_id, "error": str(exc)})

    if not vectors:
        raise RuntimeError("no valid video CLIP files were indexed")
    if errors:
        raise RuntimeError(f"dataset alignment failed for {len(errors)} videos: {errors[:3]}")

    manifest = pd.DataFrame(rows)
    matrix = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    if len(manifest) != len(matrix):
        raise RuntimeError(f"final alignment mismatch: manifest={len(manifest)}, embeddings={len(matrix)}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    np.save(embedding_path, matrix)

    report: dict[str, object] = {
        "videos_indexed": len(clip_files),
        "rows": len(manifest),
        "dimension": int(matrix.shape[1]),
        "source_dtype": "float16",
        "index_dtype": "float32",
        "object_rows_resolved": int(object_hits),
        "object_rows_missing": int(len(manifest) - object_hits),
        "clip_dir": str(clip_root),
        "mapping_dir": str(mapping_root),
        "keyframes_dir": str(keyframe_root),
        "objects_dir": str(objects_dir) if objects_dir else "",
        "manifest": str(manifest_path),
        "embeddings": str(embedding_path),
    }
    if report_output is not None:
        report_path = Path(report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
