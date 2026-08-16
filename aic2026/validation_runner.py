from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .evidence import SQLiteEvidenceStore
from .pipeline import AICPipeline
from .query_encoder import CLIPQueryEncoder
from .query_manifest import load_query_manifest
from .retrieval import FrameIndex
from .validation_package import write_validation_package


def run_validation_package(
    queries_path: str | Path,
    manifest_path: str | Path,
    embeddings_path: str | Path,
    videos_dir: str | Path,
    output_dir: str | Path,
    *,
    media_info_dir: str | Path | None = None,
    faiss_index_path: str | Path | None = None,
    asr_db: str | Path | None = None,
    ocr_db: str | Path | None = None,
    caption_db: str | Path | None = None,
    evidence_rrf_k: int = 60,
    top_k: int = 100,
    radius_frames: int = 24,
    max_decode_frames: int = 96,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
    fine_score: bool = True,
    query_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Generate ranked candidates for human validation from the supplied workbook.

    No ground truth is loaded or inferred here. The function only produces model
    predictions and marks the resulting package as lacking official GT.
    """
    queries = load_query_manifest(queries_path)
    if query_ids:
        selected = set(query_ids)
        queries = [q for q in queries if q.query_id in selected]

    if faiss_index_path:
        index = FrameIndex.from_persisted_faiss(manifest_path, embeddings_path, faiss_index_path)
    else:
        index = FrameIndex.from_files(manifest_path, embeddings_path)

    stores = []
    for path, name in ((asr_db, "asr"), (ocr_db, "ocr"), (caption_db, "caption")):
        if path:
            stores.append(SQLiteEvidenceStore(path, name))

    pipeline = AICPipeline(
        index,
        videos_dir=videos_dir,
        media_info_dir=media_info_dir,
        evidence_stores=stores,
        evidence_rrf_k=evidence_rrf_k,
    )
    encoder = CLIPQueryEncoder(model_name=model_name, pretrained=pretrained, device=device)

    grounder = None
    if fine_score:
        from .temporal_grounding import CLIPTemporalGrounder
        grounder = CLIPTemporalGrounder.create(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "official_ground_truth_available": False,
        "queries_requested": len(queries),
        "queries_completed": 0,
        "queries_failed": 0,
        "top_k": top_k,
        "fine_score": fine_score,
        "model_name": model_name,
        "pretrained": pretrained,
        "device": device,
        "evidence_requested": [s.name for s in stores],
        "evidence_available": [s.name for s in stores if s.available],
        "outputs": [],
        "failures": [],
    }

    for query in queries:
        try:
            embedding = encoder.encode_one(query.description_en or query.description_vi)
            scorer = grounder.scorer(embedding) if grounder is not None else None
            candidates = pipeline.run(
                query.description_en or query.description_vi,
                embedding,
                top_k=top_k,
                frame_scorer=scorer,
                radius_frames=radius_frames,
                max_decode_frames=max_decode_frames,
            )
            path = write_validation_package(query, candidates, output, top_k=top_k)
            summary["outputs"].append(str(path))
            summary["queries_completed"] = int(summary["queries_completed"]) + 1
        except Exception as exc:
            summary["queries_failed"] = int(summary["queries_failed"]) + 1
            summary["failures"].append({"query_id": query.query_id, "error": f"{type(exc).__name__}: {exc}"})

    report_path = output / "run_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AIC 2026 ranked candidates for human validation")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--media-info-dir", type=Path)
    parser.add_argument("--faiss-index", type=Path)
    parser.add_argument("--asr-db", type=Path)
    parser.add_argument("--ocr-db", type=Path)
    parser.add_argument("--caption-db", type=Path)
    parser.add_argument("--evidence-rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--radius-frames", type=int, default=24)
    parser.add_argument("--max-decode-frames", type=int, default=96)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-fine-score", action="store_true")
    parser.add_argument("--query-id", action="append", dest="query_ids")
    args = parser.parse_args()
    summary = run_validation_package(
        queries_path=args.queries,
        manifest_path=args.manifest,
        embeddings_path=args.embeddings,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        media_info_dir=args.media_info_dir,
        faiss_index_path=args.faiss_index,
        asr_db=args.asr_db,
        ocr_db=args.ocr_db,
        caption_db=args.caption_db,
        evidence_rrf_k=args.evidence_rrf_k,
        top_k=args.top_k,
        radius_frames=args.radius_frames,
        max_decode_frames=args.max_decode_frames,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        fine_score=not args.no_fine_score,
        query_ids=args.query_ids,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
