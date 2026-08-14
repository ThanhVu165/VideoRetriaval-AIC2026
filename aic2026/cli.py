from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="aic2026")
    sub = parser.add_subparsers(dest="command", required=True)

    video = sub.add_parser("video-manifest", help="Probe source videos and build a manifest")
    video.add_argument("--video-dir", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "video-manifest":
        from .video_manifest import build_manifest

        report = build_manifest(args.video_dir, args.output, ("mp4", "mkv", "mov", "webm"))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
