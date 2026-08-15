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

    dataset_index = sub.add_parser(
        "dataset-index", help="Build a row-aligned unified CLIP/keyframe/mapping dataset index"
    )
    dataset_index.add_argument("--clip-dir", type=Path, required=True)
    dataset_index.add_argument("--mapping-dir", type=Path, required=True)
    dataset_index.add_argument("--keyframes-dir", type=Path, required=True)
    dataset_index.add_argument("--output-manifest", type=Path, required=True)
    dataset_index.add_argument("--output-embeddings", type=Path, required=True)
    dataset_index.add_argument("--report-output", type=Path)

    build_index = sub.add_parser("build-index", help="Build a persistent FAISS frame index")
    build_index.add_argument("--manifest", type=Path, required=True)
    build_index.add_argument("--embeddings", type=Path, required=True)
    build_index.add_argument("--output-index", type=Path, required=True)
    build_index.add_argument("--metadata-output", type=Path)

    inspect_evidence = sub.add_parser("inspect-evidence", help="Inspect an ASR/OCR/caption SQLite artifact")
    inspect_evidence.add_argument("--db", type=Path, required=True)

    retrieve = sub.add_parser("retrieve", help="Run CLIP retrieval + optional multimodal evidence + temporal localization")
    retrieve.add_argument("--manifest", type=Path, required=True)
    retrieve.add_argument("--embeddings", type=Path, required=True)
    retrieve.add_argument("--faiss-index", type=Path)
    retrieve.add_argument("--videos-dir", type=Path, required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--query-embedding", type=Path)
    retrieve.add_argument("--model-name", default="ViT-B-32")
    retrieve.add_argument("--pretrained", default="openai")
    retrieve.add_argument("--device", default="cpu")
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--media-info-dir", type=Path)
    retrieve.add_argument("--asr-db", type=Path, help="Optional ASR SQLite artifact")
    retrieve.add_argument("--ocr-db", type=Path, help="Optional OCR SQLite artifact")
    retrieve.add_argument("--caption-db", type=Path, help="Optional caption SQLite artifact")
    retrieve.add_argument("--evidence-rrf-k", type=int, default=60)
    retrieve.add_argument("--top-k", type=int, default=100)
    retrieve.add_argument("--radius-frames", type=int, default=24)
    retrieve.add_argument("--max-decode-frames", type=int, default=96)

    benchmark = sub.add_parser("benchmark", help="Run a query set and report retrieval/frame metrics")
    benchmark.add_argument("--queries", type=Path, required=True)
    benchmark.add_argument("--query-column", default="Description")
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--embeddings", type=Path, required=True)
    benchmark.add_argument("--faiss-index", type=Path)
    benchmark.add_argument("--videos-dir", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--ground-truth", type=Path)
    benchmark.add_argument("--model-name", default="ViT-B-32")
    benchmark.add_argument("--pretrained", default="openai")
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--asr-db", type=Path)
    benchmark.add_argument("--ocr-db", type=Path)
    benchmark.add_argument("--caption-db", type=Path)
    benchmark.add_argument("--evidence-rrf-k", type=int, default=60)
    benchmark.add_argument("--top-k", type=int, default=100)
    benchmark.add_argument("--localize-top-k", type=int, default=0, help="0 disables video decoding during benchmark")
    benchmark.add_argument("--radius-frames", type=int, default=24)
    benchmark.add_argument("--max-decode-frames", type=int, default=96)
    benchmark.add_argument("--frame-tolerance", type=int, default=10)

    gt_template = sub.add_parser(
        "ground-truth-template",
        help="Create an annotation template from the official query spreadsheet",
    )
    gt_template.add_argument("--queries", type=Path, required=True)
    gt_template.add_argument("--query-column", default="Description")
    gt_template.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "video-manifest":
        from .video_manifest import build_manifest

        report = build_manifest(args.video_dir, args.output, ("mp4", "mkv", "mov", "webm"))
        print(json.dumps(report, indent=2))
        return

    if args.command == "dataset-index":
        from .dataset_index import build_unified_dataset

        report = build_unified_dataset(
            clip_dir=args.clip_dir,
            mapping_dir=args.mapping_dir,
            keyframes_dir=args.keyframes_dir,
            output_manifest=args.output_manifest,
            output_embeddings=args.output_embeddings,
            report_output=args.report_output,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "build-index":
        from .index_builder import build_faiss_index

        report = build_faiss_index(
            args.manifest,
            args.embeddings,
            args.output_index,
            metadata_output=args.metadata_output,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "inspect-evidence":
        from .evidence import inspect_sqlite_evidence

        print(json.dumps(inspect_sqlite_evidence(args.db), indent=2, ensure_ascii=False))
        return

    if args.command == "ground-truth-template":
        from .ground_truth import build_ground_truth_template

        report = build_ground_truth_template(
            queries_path=args.queries,
            output_path=args.output,
            query_column=args.query_column,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "benchmark":
        from .evidence import SQLiteEvidenceStore
        from .benchmark import run_benchmark

        stores = []
        if args.asr_db:
            stores.append(SQLiteEvidenceStore(args.asr_db, "asr"))
        if args.ocr_db:
            stores.append(SQLiteEvidenceStore(args.ocr_db, "ocr"))
        if args.caption_db:
            stores.append(SQLiteEvidenceStore(args.caption_db, "caption"))

        report = run_benchmark(
            queries_path=args.queries,
            manifest_path=args.manifest,
            embeddings_path=args.embeddings,
            faiss_index_path=args.faiss_index,
            videos_dir=args.videos_dir,
            output_dir=args.output_dir,
            query_column=args.query_column,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
            top_k=args.top_k,
            localize_top_k=args.localize_top_k,
            radius_frames=args.radius_frames,
            max_decode_frames=args.max_decode_frames,
            ground_truth_path=args.ground_truth,
            frame_tolerance=args.frame_tolerance,
            evidence_stores=stores,
            evidence_rrf_k=args.evidence_rrf_k,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "retrieve":
        from .evidence import SQLiteEvidenceStore
        from .pipeline import AICPipeline
        from .retrieval import FrameIndex

        if args.faiss_index:
            index = FrameIndex.from_persisted_faiss(args.manifest, args.embeddings, args.faiss_index)
        else:
            index = FrameIndex.from_files(args.manifest, args.embeddings)

        if args.query_embedding:
            query_embedding = np.load(args.query_embedding, allow_pickle=False)
        else:
            from .query_encoder import CLIPQueryEncoder

            encoder = CLIPQueryEncoder(
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
            )
            query_embedding = encoder.encode_one(args.query)

        stores = []
        if args.asr_db:
            stores.append(SQLiteEvidenceStore(args.asr_db, "asr"))
        if args.ocr_db:
            stores.append(SQLiteEvidenceStore(args.ocr_db, "ocr"))
        if args.caption_db:
            stores.append(SQLiteEvidenceStore(args.caption_db, "caption"))

        pipeline = AICPipeline(
            index,
            args.videos_dir,
            media_info_dir=args.media_info_dir,
            evidence_stores=stores,
            evidence_rrf_k=args.evidence_rrf_k,
        )
        result = pipeline.run(
            args.query,
            query_embedding,
            top_k=args.top_k,
            radius_frames=args.radius_frames,
            max_decode_frames=args.max_decode_frames,
        )
        pipeline.write_candidates(result, args.output, top_k=args.top_k)
        print(
            json.dumps(
                {
                    "rows": len(result),
                    "output": str(args.output),
                    "evidence_modalities": [s.name for s in stores if s.available],
                    "evidence_requested": [s.name for s in stores],
                    "evidence_rrf_k": args.evidence_rrf_k,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
