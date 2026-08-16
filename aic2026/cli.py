from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _existing(path: Path | None) -> Path | None:
    return path if path is not None and path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(prog="aic2026")
    sub = parser.add_subparsers(dest="command", required=True)

    video = sub.add_parser("video-manifest", help="Probe source videos and build a manifest")
    video.add_argument("--video-dir", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)

    dataset_index = sub.add_parser("dataset-index", help="Build a row-aligned unified CLIP/keyframe/mapping/object dataset index")
    dataset_index.add_argument("--clip-dir", type=Path, required=True)
    dataset_index.add_argument("--mapping-dir", type=Path, required=True)
    dataset_index.add_argument("--keyframes-dir", type=Path, required=True)
    dataset_index.add_argument("--objects-dir", type=Path, default=Path("data/objects"))
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

    retrieve = sub.add_parser("retrieve", help="Run CLIP retrieval + BTC objects/metadata + optional evidence + temporal localization")
    retrieve.add_argument("--manifest", type=Path, required=True)
    retrieve.add_argument("--embeddings", type=Path, required=True)
    retrieve.add_argument("--faiss-index", type=Path)
    retrieve.add_argument("--videos-dir", type=Path, required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--query-translated", help="Optional English translation of the query; fused with the original text embedding")
    retrieve.add_argument("--query-embedding", type=Path)
    retrieve.add_argument("--model-name", default="ViT-B-32-quickgelu")
    retrieve.add_argument("--pretrained", default="openai")
    retrieve.add_argument("--device", default="cpu")
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--media-info-dir", type=Path, default=Path("data/media_info"))
    retrieve.add_argument("--objects-dir", type=Path, default=Path("data/objects"))
    retrieve.add_argument("--asr-db", type=Path, default=Path("artifacts/asr.sqlite"))
    retrieve.add_argument("--ocr-db", type=Path, default=Path("artifacts/ocr.sqlite"))
    retrieve.add_argument("--caption-db", type=Path, default=Path("artifacts/caption.sqlite"))
    retrieve.add_argument("--evidence-rrf-k", type=int, default=60)
    retrieve.add_argument("--beit3", action="store_true", help="Also encode the query with BEiT-3 and fuse with CLIP")
    retrieve.add_argument("--beit3-index", type=Path, default=Path("artifacts/beit3.faiss"))
    retrieve.add_argument("--beit3-weight", type=float, default=0.35)
    retrieve.add_argument("--top-k", type=int, default=100)
    retrieve.add_argument("--radius-frames", type=int, default=24)
    retrieve.add_argument("--max-decode-frames", type=int, default=96)

    benchmark = sub.add_parser("benchmark", help="Run a query set and report retrieval/frame metrics")
    benchmark.add_argument("--queries", type=Path, required=True)
    benchmark.add_argument("--query-column", default="Trans")
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--embeddings", type=Path, required=True)
    benchmark.add_argument("--faiss-index", type=Path)
    benchmark.add_argument("--videos-dir", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--ground-truth", type=Path)
    benchmark.add_argument("--model-name", default="ViT-B-32-quickgelu")
    benchmark.add_argument("--pretrained", default="openai")
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--asr-db", type=Path, default=Path("artifacts/asr.sqlite"))
    benchmark.add_argument("--ocr-db", type=Path, default=Path("artifacts/ocr.sqlite"))
    benchmark.add_argument("--caption-db", type=Path, default=Path("artifacts/caption.sqlite"))
    benchmark.add_argument("--evidence-rrf-k", type=int, default=60)
    benchmark.add_argument("--top-k", type=int, default=100)
    benchmark.add_argument("--localize-top-k", type=int, default=0)
    benchmark.add_argument("--radius-frames", type=int, default=24)
    benchmark.add_argument("--max-decode-frames", type=int, default=96)
    benchmark.add_argument("--frame-tolerance", type=int, default=10)

    gt_template = sub.add_parser("ground-truth-template", help="Create an annotation template from the official query spreadsheet")
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
            clip_dir=args.clip_dir, mapping_dir=args.mapping_dir, keyframes_dir=args.keyframes_dir,
            output_manifest=args.output_manifest, output_embeddings=args.output_embeddings,
            report_output=args.report_output, objects_dir=args.objects_dir,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "build-index":
        from .index_builder import build_faiss_index
        report = build_faiss_index(args.manifest, args.embeddings, args.output_index, metadata_output=args.metadata_output)
        print(json.dumps(report, indent=2))
        return

    if args.command == "inspect-evidence":
        from .evidence import inspect_sqlite_evidence
        print(json.dumps(inspect_sqlite_evidence(args.db), indent=2, ensure_ascii=False))
        return

    if args.command == "ground-truth-template":
        from .ground_truth import build_ground_truth_template
        report = build_ground_truth_template(queries_path=args.queries, output_path=args.output, query_column=args.query_column)
        print(json.dumps(report, indent=2))
        return

    if args.command == "benchmark":
        from .evidence import SQLiteEvidenceStore
        from .benchmark import run_benchmark
        stores = []
        for path, name in ((args.asr_db, "asr"), (args.ocr_db, "ocr"), (args.caption_db, "caption")):
            p = _existing(path)
            if p:
                stores.append(SQLiteEvidenceStore(p, name))
        report = run_benchmark(
            queries_path=args.queries, manifest_path=args.manifest, embeddings_path=args.embeddings,
            faiss_index_path=args.faiss_index, videos_dir=args.videos_dir, output_dir=args.output_dir,
            query_column=args.query_column, model_name=args.model_name, pretrained=args.pretrained,
            device=args.device, top_k=args.top_k, localize_top_k=args.localize_top_k,
            radius_frames=args.radius_frames, max_decode_frames=args.max_decode_frames,
            ground_truth_path=args.ground_truth, frame_tolerance=args.frame_tolerance,
            evidence_stores=stores, evidence_rrf_k=args.evidence_rrf_k,
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
            encoder = CLIPQueryEncoder(model_name=args.model_name, pretrained=args.pretrained, device=args.device)
            query_embedding = encoder.encode_one(args.query)
            if args.query_translated:
                translated = encoder.encode_one(args.query_translated)
                query_embedding = 0.20 * query_embedding + 0.80 * translated
                query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)

        stores = []
        for path, name in ((args.asr_db, "asr"), (args.ocr_db, "ocr"), (args.caption_db, "caption")):
            p = _existing(path)
            if p:
                stores.append(SQLiteEvidenceStore(p, name))

        beit3_index = None
        beit3_query_embedding = None
        if args.beit3:
            from .beit3 import BEiT3Index
            from .beit3_query import BEiT3QueryEncoder
            beit3_index = BEiT3Index.from_files(args.manifest, args.beit3_index)
            beit3_encoder = BEiT3QueryEncoder(device=args.device)
            beit3_query_embedding = beit3_encoder.encode_one(args.query_translated or args.query)

        pipeline = AICPipeline(
            index, args.videos_dir, media_info_dir=_existing(args.media_info_dir), objects_dir=_existing(args.objects_dir),
            evidence_stores=stores, evidence_rrf_k=args.evidence_rrf_k,
            beit3_index=beit3_index, beit3_weight=args.beit3_weight,
        )
        result = pipeline.run(
            args.query, query_embedding, top_k=args.top_k,
            radius_frames=args.radius_frames, max_decode_frames=args.max_decode_frames,
            beit3_query_embedding=beit3_query_embedding,
        )
        pipeline.write_candidates(result, args.output, top_k=args.top_k)
        print(json.dumps({
            "rows": len(result), "output": str(args.output),
            "evidence_modalities": [s.name for s in stores if s.available],
            "evidence_requested": [s.name for s in stores],
            "btc_metadata_enabled": bool(_existing(args.media_info_dir)),
            "btc_objects_enabled": bool(_existing(args.objects_dir)),
            "translated_query_enabled": bool(args.query_translated),
            "evidence_rrf_k": args.evidence_rrf_k,
            "beit3_enabled": bool(args.beit3),
            "beit3_weight": args.beit3_weight if args.beit3 else 0.0,
        }, indent=2))


if __name__ == "__main__":
    main()
