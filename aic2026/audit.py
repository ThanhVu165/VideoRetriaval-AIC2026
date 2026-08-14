from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
JSON_SUFFIXES = {".json"}
FEATURE_SUFFIXES = {".npy", ".npz"}
CSV_SUFFIXES = {".csv"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_files(root: Path, suffixes: set[str] | None = None):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes):
            yield p


def audit_images(root: Path) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for p in iter_files(root, IMAGE_SUFFIXES):
        video_id = p.parent.name
        counts[video_id] += 1
        rows.append({
            "video_id": video_id,
            "keyframe_name": p.stem,
            "keyframe_path": str(p),
            "suffix": p.suffix.lower(),
        })
    return rows, counts


def inspect_feature_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in iter_files(root, FEATURE_SUFFIXES):
        item: dict[str, Any] = {"path": str(p), "suffix": p.suffix.lower()}
        try:
            if p.suffix.lower() == ".npy":
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                item.update({"shape": list(arr.shape), "dtype": str(arr.dtype), "ndim": arr.ndim})
            else:
                with np.load(p, allow_pickle=False) as z:
                    item["arrays"] = {
                        k: {"shape": list(v.shape), "dtype": str(v.dtype), "ndim": v.ndim}
                        for k, v in z.items()
                    }
        except Exception as exc:
            item["error"] = repr(exc)
        rows.append(item)
    return rows


def inspect_json_files(root: Path) -> dict[str, Any]:
    files = list(iter_files(root, JSON_SUFFIXES)) if root.exists() else []
    top_keys: Counter = Counter()
    samples: list[dict[str, Any]] = []
    for p in files[:10]:
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            keys = list(obj.keys()) if isinstance(obj, dict) else []
            top_keys.update(keys)
            samples.append({"path": str(p), "type": type(obj).__name__, "keys": keys[:20]})
        except Exception as exc:
            samples.append({"path": str(p), "error": repr(exc)})
    return {"count": len(files), "samples": samples, "top_level_keys": dict(top_keys)}


def inspect_mapping_files(root: Path) -> dict[str, Any]:
    """Inspect BTC keyframe mapping CSVs without assuming their column names."""
    files = list(iter_files(root, CSV_SUFFIXES)) if root.exists() else []
    column_counts: Counter = Counter()
    samples: list[dict[str, Any]] = []

    for p in files[:10]:
        try:
            df = pd.read_csv(p)
            columns = [str(c) for c in df.columns]
            column_counts.update(columns)
            sample_rows = df.head(3).to_dict(orient="records")
            samples.append({
                "path": str(p),
                "rows": int(len(df)),
                "columns": columns,
                "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
                "sample_rows": sample_rows,
            })
        except Exception as exc:
            samples.append({"path": str(p), "error": repr(exc)})

    return {
        "count": len(files),
        "samples": samples,
        "column_frequency": dict(column_counts),
    }


def audit(config_path: str) -> dict[str, Any]:
    cfg = load_config(Path(config_path))
    keyframes_dir = Path(cfg["keyframes_dir"])
    clip_dir = Path(cfg["clip_dir"])
    mapping_dir = Path(cfg["mapping_dir"])
    media_info_dir = Path(cfg["media_info_dir"])
    objects_dir = Path(cfg.get("objects_dir", ""))
    artifacts_dir = Path(cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    image_rows, counts = audit_images(keyframes_dir)
    summary = {
        "keyframes": {
            "total": len(image_rows),
            "videos": len(counts),
            "top_videos_by_keyframes": counts.most_common(20),
        },
        "clip_features": inspect_feature_files(clip_dir),
        "mapping": inspect_mapping_files(mapping_dir),
        "media_info": inspect_json_files(media_info_dir),
        "objects": inspect_json_files(objects_dir) if objects_dir else {"count": 0, "samples": [], "top_level_keys": {}},
        "notes": [
            "BTC keyframe mappings are inspected as CSV files.",
            "The audit recursively scans nested archive wrapper directories.",
            "Original frame IDs will be normalized after the mapping schema is confirmed.",
        ],
    }

    max_rows = int(cfg.get("audit", {}).get("max_manifest_rows", 5_000_000))
    manifest = pd.DataFrame(image_rows[:max_rows])
    if manifest.empty:
        manifest = pd.DataFrame(columns=["video_id", "keyframe_name", "keyframe_path", "suffix"])
    manifest.to_csv(artifacts_dir / "dataset_manifest.csv", index=False)

    with (artifacts_dir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    with (artifacts_dir / "audit_report.txt").open("w", encoding="utf-8") as f:
        f.write("AIC 2026 Dataset Audit\n")
        f.write("=======================\n")
        f.write(f"Total keyframes: {len(image_rows):,}\n")
        f.write(f"Videos represented by keyframes: {len(counts):,}\n")
        f.write(f"CLIP feature files: {len(summary['clip_features'])}\n")
        f.write(f"Mapping CSV files: {summary['mapping']['count']}\n")
        f.write(f"Media-info JSON files: {summary['media_info']['count']}\n")
        f.write(f"Object JSON files: {summary['objects']['count']}\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AIC 2026 dataset layout and feature files.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
