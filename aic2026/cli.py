from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(prog="aic2026")
    sub = parser.add_subparsers(dest="command", required=True)

    video = sub.add_parser("video-manifest", help="Probe source videos and build a manifest")
    video.add_argument("--video-dir", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)

    retrieve = sub.add_parser("retrieve", help="Run CLIP retrieval + temporal localization")
    retrieve.add_argument("--manifest", type=Path, required=True)
    retrieve.add_argument("--embeddings", type=Path, required=True)
    retrieve.add_argument("--videos-dir", type=Path, required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--query-embedding", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--media-info-dir", type=Path)
    retrieve.add_argument("--top-k", type=int, default=100)
    retrieve.add_argument("--radius-frames", type=int, default=15)

    args = parser.parse_args()
    if args.command == "video-manifest":
        from .video_manifest import build_manifest

        report = build_manifest(args.video_dir, args.output, ("mp4", "mkv", "mov", "webm"))
        print(json.dumps(report, indent=2))
        return

    if args.command == "retrieve":
        from .pipeline import AICPipeline
        from .retrieval import FrameIndex

        index = FrameIndex.from_files(args.manifest, args.embeddings)
        query_embedding = np.load(args.query_embedding, allow_pickle=False)
        pipeline = AICPipeline(index, args.videos_dir, media_info_dir=args.media_info_dir)
        result = pipeline.run(
            args.query,
            query_embedding,
            top_k=args.top_k,
            radius_frames=args.radius_frames,
        )
        pipeline.write_candidates(result, args.output, top_k=args.top_k)
        print(json.dumps({"rows": len(result), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
