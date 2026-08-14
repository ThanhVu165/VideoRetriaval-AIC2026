from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aic2026.video import probe_video


def discover_videos(video_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    videos: list[Path] = []
    for ext in extensions:
        videos.extend(video_dir.glob(f"*.{ext}"))
    return sorted(set(videos))


def build_manifest(video_dir: Path, output: Path, extensions: tuple[str, ...]) -> dict[str, int]:
    videos = discover_videos(video_dir, extensions)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for path in videos:
        try:
            info = probe_video(path)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": str(exc)})
            continue
        rows.append(
            {
                "video_id": info.video_id,
                "video_path": info.path,
                "fps": info.fps,
                "frame_count": info.frame_count,
                "duration_sec": info.duration_sec,
                "width": info.width,
                "height": info.height,
            }
        )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "video_id", "video_path", "fps", "frame_count", "duration_sec", "width", "height"
        ])
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "videos_found": len(videos),
        "videos_indexed": len(rows),
        "errors": errors,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"videos_found": len(videos), "videos_indexed": len(rows), "errors": len(errors)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source-video manifest for AIC temporal localization.")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extensions", nargs="+", default=["mp4", "mkv", "mov", "webm"])
    args = parser.parse_args()

    report = build_manifest(args.video_dir, args.output, tuple(args.extensions))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
