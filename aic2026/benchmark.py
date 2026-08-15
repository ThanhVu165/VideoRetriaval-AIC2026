from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import mean_recall_at_k, reciprocal_rank
from .pipeline import RetrievalPipeline
from .retrieval import FrameIndex


DEFAULT_KS = (1, 5, 20, 50, 100)


def load_queries(path: str | Path, query_column: str = "Description") -> pd.DataFrame:
    """Load query IDs and text from xlsx/csv/json without assuming a fixed dataset schema."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(payload if isinstance(payload, list) else payload["queries"])
    else:
        raise ValueError(f"Unsupported query file format: {path.suffix}")

    id_candidates = ["Query Name", "query_id", "id", "name"]
    qid = next((c for c in id_candidates if c in df.columns), None)
    if qid is None:
        raise ValueError(f"No query ID column found. Columns: {df.columns.tolist()}")
    if query_column not in df.columns:
        raise ValueError(f"Query column {query_column!r} not found. Columns: {df.columns.tolist()}")

    out = pd.DataFrame({"query_id": df[qid].astype(str), "query": df[query_column].fillna("").astype(str)})
    out = out[out["query"].str.strip().ne("")].drop_duplicates("query_id").reset_index(drop=True)
    return out


def load_ground_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load explicit GT JSON.

    Supported record form:
      {"query_id": {"video_ids": ["L21_V001"], "frames": {"L21_V001": [1234]}}}
    or a list of records with query_id/video_ids/frames.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("queries", payload)
    if isinstance(records, dict):
        return {str(k): dict(v) for k, v in records.items()}
    result: dict[str, dict[str, Any]] = {}
    for item in records:
        qid = str(item["query_id"])
        result[qid] = {
            "video_ids": [str(x) for x in item.get("video_ids", [])],
            "frames": item.get("frames", {}),
            "intervals": item.get("intervals", {}),
        }
    return result


def _ranked_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(x["video_id"]) for x in rows]


def _frame_hit(row: dict[str, Any], gt: dict[str, Any], tolerance: int) -> bool:
    video_id = str(row.get("video_id", ""))
    frames = gt.get("frames", {}).get(video_id, [])
    pred = row.get("semantic_keyframe", row.get("best_frame_id"))
    if pred is None or not frames:
        return False
    return any(abs(int(pred) - int(frame)) <= tolerance for frame in frames)


def evaluate_results(
    results: dict[str, list[dict[str, Any]]],
    ground_truth: dict[str, dict[str, Any]] | None,
    ks: tuple[int, ...] = DEFAULT_KS,
    frame_tolerance: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for query_id, rows in results.items():
        gt = (ground_truth or {}).get(query_id, {})
        relevant = set(str(x) for x in gt.get("video_ids", []))
        ranked = _ranked_ids(rows)
        row: dict[str, Any] = {
            "query_id": query_id,
            "num_candidates": len(rows),
            "top1_video_id": ranked[0] if ranked else "",
            "top1_score": float(rows[0]["rank_score"]) if rows else None,
            "top2_score": float(rows[1]["rank_score"]) if len(rows) > 1 else None,
            "top1_top2_gap": (float(rows[0]["rank_score"]) - float(rows[1]["rank_score"])) if len(rows) > 1 else None,
            "gt_available": bool(relevant),
        }
        if relevant:
            row["mrr"] = reciprocal_rank(ranked, relevant)
            for k in ks:
                row[f"r@{k}"] = float(bool(set(ranked[:k]) & relevant))
            frame_hits = [
                any(_frame_hit(candidate, gt, frame_tolerance) for candidate in rows[:k])
                for k in ks
            ]
            for k, hit in zip(ks, frame_hits):
                row[f"frame_r@{k}"] = float(hit)
        else:
            row["mrr"] = None
            for k in ks:
                row[f"r@{k}"] = None
                row[f"frame_r@{k}"] = None
        per_query.append(row)

    if not per_query:
        return {"queries": 0}, []

    df = pd.DataFrame(per_query)
    summary: dict[str, Any] = {
        "queries": int(len(df)),
        "queries_with_ground_truth": int(df["gt_available"].sum()),
        "mean_top1_top2_gap": float(df["top1_top2_gap"].dropna().mean()) if df["top1_top2_gap"].notna().any() else None,
    }
    gt_df = df[df["gt_available"]]
    if not gt_df.empty:
        summary["mrr"] = float(gt_df["mrr"].mean())
        for k in ks:
            summary[f"r@{k}"] = float(gt_df[f"r@{k}"].mean())
            summary[f"frame_r@{k}"] = float(gt_df[f"frame_r@{k}"].mean())
    else:
        summary["mrr"] = None
        for k in ks:
            summary[f"r@{k}"] = None
            summary[f"frame_r@{k}"] = None
    return summary, per_query


def run_benchmark(
    queries_path: str | Path,
    manifest_path: str | Path,
    embeddings_path: str | Path,
    faiss_index_path: str | Path | None,
    videos_dir: str | Path,
    output_dir: str | Path,
    query_column: str = "Description",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
    top_k: int = 100,
    localize_top_k: int = 0,
    radius_frames: int = 24,
    max_decode_frames: int = 96,
    ground_truth_path: str | Path | None = None,
    frame_tolerance: int = 10,
) -> dict[str, Any]:
    queries = load_queries(queries_path, query_column=query_column)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if faiss_index_path:
        index = FrameIndex.from_persisted_faiss(manifest_path, embeddings_path, faiss_index_path)
    else:
        index = FrameIndex.from_files(manifest_path, embeddings_path)

    from .query_encoder import CLIPQueryEncoder

    encoder = CLIPQueryEncoder(model_name=model_name, pretrained=pretrained, device=device)
    pipeline = RetrievalPipeline(index, videos_dir)

    all_results: dict[str, list[dict[str, Any]]] = {}
    for _, q in queries.iterrows():
        qid = str(q["query_id"])
        text = str(q["query"])
        embedding = encoder.encode_one(text)
        candidates = pipeline.retrieve(text, embedding, top_k_frames=max(top_k * 10, 1000), top_k_videos=top_k)
        if localize_top_k > 0 and not candidates.empty:
            localized: list[dict[str, Any]] = []
            for _, candidate in candidates.head(localize_top_k).iterrows():
                event = pipeline.localize(candidate, radius_frames=radius_frames, max_decode_frames=max_decode_frames)
                localized.append({**candidate.to_dict(), "temporal_start_frame": event.start_frame, "temporal_end_frame": event.end_frame, "semantic_keyframe": event.semantic_keyframe, "temporal_score": event.score})
            localized_df = pd.DataFrame(localized)
            tail = candidates.iloc[localize_top_k:].copy()
            all_df = pd.concat([localized_df, tail], ignore_index=True)
            candidates = all_df
        rows = candidates.to_dict(orient="records")
        all_results[qid] = rows
        (output / f"{qid}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    gt = load_ground_truth(ground_truth_path) if ground_truth_path else None
    summary, per_query = evaluate_results(all_results, gt, frame_tolerance=frame_tolerance)
    (output / "results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "per_query.json").write_text(json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
