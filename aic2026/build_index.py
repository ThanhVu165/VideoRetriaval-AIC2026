from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a contiguous frame-level CLIP matrix from Phase 0 manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output .npy file for the ordered frame matrix")
    parser.add_argument("--report", type=Path, help="Optional JSON report")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    required = {"video_id", "keyframe_idx", "clip_idx", "clip_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")

    manifest = manifest.sort_values(["video_id", "keyframe_idx"]).reset_index(drop=True)
    chunks: list[np.ndarray] = []
    errors: list[str] = []

    for video_id, group in manifest.groupby("video_id", sort=False):
        clip_path = Path(str(group.iloc[0].clip_path))
        if not clip_path.is_absolute():
            clip_path = args.clip_dir / clip_path.name
        if not clip_path.exists():
            errors.append(f"{video_id}: missing {clip_path}")
            continue

        arr = np.load(clip_path, mmap_mode="r", allow_pickle=False)
        indices = group["clip_idx"].to_numpy(dtype=np.int64)
        if indices.min(initial=0) < 0 or indices.max(initial=-1) >= arr.shape[0]:
            errors.append(f"{video_id}: clip_idx outside feature matrix")
            continue
        chunks.append(np.asarray(arr[indices], dtype=np.float32))

    if errors:
        raise SystemExit("Index build failed:\n" + "\n".join(errors[:50]))
    if not chunks:
        raise SystemExit("No CLIP features were loaded")

    matrix = np.concatenate(chunks, axis=0)
    if len(matrix) != len(manifest):
        raise SystemExit(f"Ordered matrix rows {len(matrix)} != manifest rows {len(manifest)}")
    if matrix.ndim != 2 or matrix.shape[1] != 512:
        raise SystemExit(f"Expected CLIP matrix shape [N, 512], got {matrix.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix)

    report = {
        "manifest_rows": int(len(manifest)),
        "embedding_shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "videos": int(manifest.video_id.nunique()),
        "source": "BTC-provided CLIP frame features",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
