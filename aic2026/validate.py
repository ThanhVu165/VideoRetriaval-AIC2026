from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "video_id",
    "keyframe_idx",
    "keyframe_name",
    "image_path",
    "original_frame_id",
    "pts_time",
    "fps",
    "clip_idx",
    "clip_path",
    "object_path",
}


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Manifest not found: {path}"]

    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors

    if df.empty:
        errors.append("Manifest is empty")
        return errors

    if df[["video_id", "keyframe_idx"]].duplicated().any():
        errors.append("Duplicate (video_id, keyframe_idx) rows found")

    if df["original_frame_id"].isna().any():
        errors.append("Missing original_frame_id values")

    if df["pts_time"].isna().any():
        errors.append("Missing pts_time values")

    for video_id, group in df.groupby("video_id", sort=False):
        frames = group["original_frame_id"].to_numpy()
        times = group["pts_time"].to_numpy()
        if len(frames) > 1 and (frames[1:] < frames[:-1]).any():
            errors.append(f"{video_id}: original_frame_id is not monotonic")
        if len(times) > 1 and (times[1:] < times[:-1]).any():
            errors.append(f"{video_id}: pts_time is not monotonic")

        if not (group["keyframe_idx"].to_numpy() == range(len(group))).all():
            errors.append(f"{video_id}: keyframe_idx does not start at 0 or is not contiguous")

        if not (group["clip_idx"].to_numpy() == group["keyframe_idx"].to_numpy()).all():
            errors.append(f"{video_id}: clip_idx does not align with keyframe_idx")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an AIC 2026 unified dataset manifest.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    errors = validate_manifest(Path(args.manifest))
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
