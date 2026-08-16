from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aic2026.evidence import SQLiteEvidenceStore
from aic2026.pipeline import AICPipeline
from aic2026.query_encoder import CLIPQueryEncoder
from aic2026.retrieval import FrameIndex


def main() -> None:
    p = argparse.ArgumentParser(description="Run the canonical AIC2026 retrieval pipeline over the organizer query sheet.")
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--faiss-index", type=Path)
    p.add_argument("--videos-dir", type=Path, required=True)
    p.add_argument("--media-info-dir", type=Path, default=Path("data/media_info"))
    p.add_argument("--objects-dir", type=Path, default=Path("data/objects"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--query-column", default="Description")
    p.add_argument("--translated-query-column", default="Trans")
    p.add_argument("--task-prefix", action="append", default=[])
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--radius-frames", type=int, default=24)
    p.add_argument("--max-decode-frames", type=int, default=96)
    p.add_argument("--fine-score", action="store_true")
    p.add_argument("--fine-batch-size", type=int, default=32)
    p.add_argument("--model-name", default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--device", default="cpu")
    p.add_argument("--asr-db", type=Path, default=Path("artifacts/asr.sqlite"))
    p.add_argument("--ocr-db", type=Path, default=Path("artifacts/ocr.sqlite"))
    p.add_argument("--caption-db", type=Path, default=Path("artifacts/caption.sqlite"))
    p.add_argument("--evidence-rrf-k", type=int, default=60)
    args = p.parse_args()

    queries = pd.read_excel(args.queries) if args.queries.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(args.queries)
    required = {"Query Name", args.query_column}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"query file missing columns: {sorted(missing)}")
    if args.translated_query_column and args.translated_query_column not in queries.columns:
        args.translated_query_column = ""
    if args.task_prefix:
        prefixes = tuple(args.task_prefix)
        queries = queries[queries["Query Name"].astype(str).str.startswith(prefixes)]

    args.top_k = min(max(args.top_k, 1), 100)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = FrameIndex.from_persisted_faiss(args.manifest, args.embeddings, args.faiss_index) if args.faiss_index else FrameIndex.from_files(args.manifest, args.embeddings)
    encoder = CLIPQueryEncoder(model_name=args.model_name, pretrained=args.pretrained, device=args.device)

    stores = []
    for path, name in ((args.asr_db, "asr"), (args.ocr_db, "ocr"), (args.caption_db, "caption")):
        if path.exists():
            stores.append(SQLiteEvidenceStore(path, name))

    pipeline = AICPipeline(
        index,
        args.videos_dir,
        media_info_dir=args.media_info_dir if args.media_info_dir.exists() else None,
        objects_dir=args.objects_dir if args.objects_dir.exists() else None,
        evidence_stores=stores,
        evidence_rrf_k=args.evidence_rrf_k,
    )

    for _, row in queries.iterrows():
        qid = str(row["Query Name"])
        query = str(row[args.query_column])
        translated = str(row[args.translated_query_column]) if args.translated_query_column and pd.notna(row[args.translated_query_column]) else ""
        embedding = encoder.encode_one(query)
        scoring_query = query
        if translated.strip():
            translated_embedding = encoder.encode_one(translated)
            embedding = 0.20 * embedding + 0.80 * translated_embedding
            embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
            scoring_query = translated

        scorer = None
        if args.fine_score:
            from aic2026.temporal_grounding import CLIPTemporalGrounder
            scorer = CLIPTemporalGrounder.create(
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
                batch_size=args.fine_batch_size,
            ).scorer(embedding)

        result = pipeline.run(
            query,
            embedding,
            top_k=args.top_k,
            frame_scorer=scorer,
            radius_frames=args.radius_frames,
            max_decode_frames=args.max_decode_frames,
            scoring_query=scoring_query,
        )
        pipeline.write_candidates(result, args.output_dir / f"{qid}.json", top_k=args.top_k)

    print(json.dumps({
        "queries": int(len(queries)),
        "output_dir": str(args.output_dir),
        "top_k": args.top_k,
        "translated_query": bool(args.translated_query_column),
        "btc_objects": args.objects_dir.exists(),
        "btc_metadata": args.media_info_dir.exists(),
        "evidence_modalities": [s.name for s in stores if s.available],
        "fine_score": args.fine_score,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
