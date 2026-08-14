from __future__ import annotations

import argparse
import csv
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
        return yaml.safe_load(f) or {}


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
        rows.append(
            {
                "video_id": video_id,
                "keyframe_name": p.stem,
                "keyframe_path": str(p),
                "suffix": p.suffix.lower(),
            }
        )
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
    files = list(iter_files(root, CSV_SUFFIXES)) if root.exists() else []
    column_counts: Counter = Counter()
    samples: list[dict[str, Any]] = []
    for p in files[:10]:
        try:
            df = pd.read_csv(p)
            columns = [str(c) for c in df.columns]
            column_counts.update(columns)
            samples.append(
                {
                    "path": str(p),
                    "rows": int(len(df)),
                    "columns": columns,
                    "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
                    "sample_rows": df.head(3).to_dict(orient="records"),
                }
            )
        except Exception as exc:
            samples.append({"path": str(p), "error": repr(exc)})
    return {"count": len(files), "samples": samples, "column_frequency": dict(column_counts)}


def _video_key_from_path(path: Path) -> str:
    return path.stem


def _read_mapping_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {str(c).strip().lower(): c for c in df.columns}
    required = {"n", "pts_time", "fps", "frame_idx"}
    if not required.issubset(rename_map):
        raise ValueError(f"{path}: expected mapping columns {sorted(required)}, got {list(df.columns)}")
    return df.rename(columns={
        rename_map["n"]: "n",
        rename_map["pts_time"]: "pts_time",
        rename_map["fps"]: "fps",
        rename_map["frame_idx"]: "frame_idx",
    })


def build_manifest(
    keyframes_dir: Path,
    clip_dir: Path,
    mapping_dir: Path,
    objects_dir: Path,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    image_rows, image_counts = audit_images(keyframes_dir)
    frame_rows = pd.DataFrame(image_rows[:max_rows])
    if frame_rows.empty:
        return frame_rows, {"errors": ["No keyframes found."], "videos_checked": 0}

    errors: list[str] = []
    records: list[dict[str, Any]] = []
    grouped = frame_rows.groupby("video_id", sort=True)
    clip_files = {p.stem: p for p in iter_files(clip_dir, FEATURE_SUFFIXES)}
    mapping_files = {p.stem: p for p in iter_files(mapping_dir, CSV_SUFFIXES)}

    for video_id, group in grouped:
        image_count = len(group)
        clip_path = clip_files.get(video_id)
        mapping_path = mapping_files.get(video_id)
        if clip_path is None:
            errors.append(f"{video_id}: missing CLIP feature file")
            continue
        if mapping_path is None:
            errors.append(f"{video_id}: missing mapping CSV")
            continue
        try:
            mapping = _read_mapping_file(mapping_path)
            clip = np.load(clip_path, mmap_mode="r", allow_pickle=False)
            if clip.ndim != 2:
                raise ValueError(f"CLIP array ndim={clip.ndim}, expected 2")
            if clip.shape[0] != image_count:
                errors.append(f"{video_id}: keyframes={image_count}, mapping={len(mapping)}, clip_rows={clip.shape[0]}")
            if len(mapping) != image_count:
                errors.append(f"{video_id}: keyframes={image_count}, mapping_rows={len(mapping)}")

            mapping = mapping.reset_index(drop=True)
            group = group.sort_values("keyframe_name", key=lambda s: s.str.extract(r"(\\d+)$")[0].astype(int)).reset_index(drop=True)
            rows = min(len(group), len(mapping), clip.shape[0])
            for i in range(rows):
                image_row = group.iloc[i]
                map_row = mapping.iloc[i]
                object_path = ""
                for ext in JSON_SUFFIXES:
                    candidate = objects_dir / video_id / f"{image_row['keyframe_name']}{ext}"
                    if candidate.exists():
                        object_path = str(candidate)
                        break
                records.append({
                    "video_id": video_id,
                    "keyframe_idx": i,
                    "keyframe_name": image_row["keyframe_name"],
                    "image_path": image_row["keyframe_path"],
                    "original_frame_id": int(map_row["frame_idx"]),
                    "pts_time": float(map_row["pts_time"]),
                    "fps": float(map_row["fps"]),
                    "clip_idx": i,
                    "clip_path": str(clip_path),
                    "object_path": object_path,
                })
        except Exception as exc:
            errors.append(f"{video_id}: {exc!r}")

    manifest = pd.DataFrame(records)
    if not manifest.empty:
        manifest = manifest.sort_values(["video_id", "keyframe_idx"]).reset_index(drop=True)

    checks = {
        "videos_checked": int(len(grouped)),
        "videos_with_manifest": int(manifest["video_id"].nunique()) if not manifest.empty else 0,
        "keyframes_found": int(len(frame_rows)),
        "manifest_rows": int(len(manifest)),
        "object_matches": int((manifest["object_path"] != "").sum()) if not manifest.empty else 0,
        "errors": errors,
    }
    return manifest, checks


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
    summary: dict[str, Any] = {
        "keyframes": {
            "total": len(image_rows),
            "videos": len(counts),
            "top_videos_by_keyframes": counts.most_common(20),
        },
        "clip_features": inspect_feature_files(clip_dir),
        "mapping": inspect_mapping_files(mapping_dir),
        "media_info": inspect_json_files(media_info_dir),
        "objects": inspect_json_files(objects_dir) if objects_dir else {"count": 0, "samples": [], "top_level_keys": {}},
    }

    max_rows = int(cfg.get("audit", {}).get("max_manifest_rows", 5_000_000))
    manifest, integrity = build_manifest(keyframes_dir, clip_dir, mapping_dir, objects_dir, max_rows)
    summary["integrity"] = integrity

    manifest.to_csv(artifacts_dir / "dataset_manifest.csv", index=False)
    manifest.to_parquet(artifacts_dir / "dataset_manifest.parquet", index=False)
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
        f.write(f"Manifest rows: {integrity['manifest_rows']:,}\n")
        f.write(f"Object-path matches: {integrity['object_matches']:,}\n")
        f.write(f"Integrity errors: {len(integrity['errors'])}\n")
        for error in integrity["errors"][:100]:
            f.write(f"- {error}\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and validate AIC 2026 dataset layout.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
