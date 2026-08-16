from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .beit3 import BEiT3Index
from .evidence import EvidenceStore
from .fine_scoring import FrameScorerProtocol, TemporalScoreConfig, select_peak_frame, temporal_event_score
from .multimodal import MultimodalReranker
from .ranking import RankingWeights, rerank_candidates, top_k_submission
from .retrieval import FrameIndex
from .support_data import load_metadata_text, resolve_object_path
from .temporal import merge_frame_hits
from .video import iter_frame_ids, probe_video

FrameScorerFn = Callable[[Sequence[object]], Sequence[float]]
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}


@dataclass(frozen=True)
class LocalizedEvent:
    video_id: str
    coarse_frame_id: int
    start_frame: int
    end_frame: int
    semantic_keyframe: int
    score: float


class VideoResolver:
    """Resolve source video IDs recursively and expose duplicate-ID diagnostics."""

    def __init__(self, videos_dir: str | Path):
        self.videos_dir = Path(videos_dir)
        self._paths: dict[str, Path] = {}
        self._duplicates: dict[str, list[Path]] = {}
        if self.videos_dir.exists():
            for path in self.videos_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                video_id = path.stem
                if video_id in self._paths:
                    self._duplicates.setdefault(video_id, [self._paths[video_id]]).append(path)
                else:
                    self._paths[video_id] = path

    @property
    def count(self) -> int:
        return len(self._paths)

    @property
    def duplicates(self) -> dict[str, list[Path]]:
        return self._duplicates

    def resolve(self, video_id: str) -> Path | None:
        return self._paths.get(video_id)


class RetrievalPipeline:
    def __init__(self, frame_index: FrameIndex, videos_dir: str | Path, media_info_dir: str | Path | None = None,
                 objects_dir: str | Path | None = None, ranking_weights: RankingWeights | None = None,
                 temporal_config: TemporalScoreConfig | None = None, evidence_stores: Sequence[EvidenceStore] | None = None,
                 evidence_rrf_k: int = 60, beit3_index: BEiT3Index | None = None, beit3_weight: float = 0.35):
        self.frame_index = frame_index
        self.videos_dir = Path(videos_dir)
        self.media_info_dir = Path(media_info_dir) if media_info_dir else None
        self.objects_dir = Path(objects_dir) if objects_dir else None
        self.video_resolver = VideoResolver(self.videos_dir)
        self.reranker = MultimodalReranker(evidence_stores=evidence_stores, rrf_k=evidence_rrf_k)
        self.ranking_weights = ranking_weights or RankingWeights(retrieval=0.55, multimodal=0.30, temporal=0.15)
        self.temporal_config = temporal_config or TemporalScoreConfig()
        self.beit3_index = beit3_index
        if not 0.0 <= beit3_weight <= 1.0:
            raise ValueError("beit3_weight must be in [0, 1]")
        self.beit3_weight = float(beit3_weight)

    def _enrich_support_paths(self, candidates: pd.DataFrame) -> pd.DataFrame:
        out = candidates.copy()
        if "object_path" not in out:
            out["object_path"] = ""
        resolved: list[str] = []
        for _, row in out.iterrows():
            current = str(row.get("object_path", "") or "")
            if current and Path(current).is_file():
                resolved.append(current)
                continue
            resolved.append("")
        if any(not x for x in resolved):
            for i, row in out.iterrows():
                if resolved[i]:
                    continue
                matches = self.frame_index.manifest[
                    (self.frame_index.manifest["video_id"].astype(str) == str(row["video_id"]))
                    & (self.frame_index.manifest["original_frame_id"].astype(int) == int(row["best_frame_id"]))
                ]
                if not matches.empty:
                    image_path = str(matches.iloc[0]["image_path"])
                    resolved[i] = resolve_object_path(self.objects_dir, str(row["video_id"]), image_path)
        out["object_path"] = resolved
        out["metadata_text"] = [load_metadata_text(self.media_info_dir, str(v)) for v in out["video_id"]]
        return out

    @staticmethod
    def _aggregate_frame_evidence(rows: pd.DataFrame, per_video_k: int = 5) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        records: list[dict[str, object]] = []
        for video_id, group in rows.groupby("video_id", sort=False):
            top = group.head(per_video_k)
            best = top.iloc[0]
            scores = top["score"].to_numpy(dtype=np.float32)
            top_mean = float(scores.mean())
            best_score = float(scores[0])
            records.append({"video_id": str(video_id), "score": 0.75 * best_score + 0.25 * top_mean,
                            "retrieval_score": 0.75 * best_score + 0.25 * top_mean,
                            "best_frame_idx": int(best["keyframe_idx"]), "best_frame_id": int(best["original_frame_id"]),
                            "best_pts_time": float(best["pts_time"]), "object_path": str(best.get("object_path", "")),
                            "retrieval_best_score": best_score, "retrieval_topk_mean": top_mean,
                            "retrieval_score_std": float(np.std(scores)) if len(scores) > 1 else 0.0})
        return pd.DataFrame(records).sort_values("retrieval_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _aggregate_beit3(rows: pd.DataFrame, per_video_k: int = 3) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame(columns=["video_id", "beit3_score", "beit3_best_frame_id", "beit3_pts_time"])
        records: list[dict[str, object]] = []
        rows = rows.sort_values("score", ascending=False)
        for video_id, group in rows.groupby("video_id", sort=False):
            top = group.head(per_video_k); best = top.iloc[0]
            records.append({"video_id": str(video_id), "beit3_score": float(0.75 * best["score"] + 0.25 * top["score"].mean()),
                            "beit3_best_frame_id": int(best["original_frame_id"]), "beit3_pts_time": float(best["pts_time"])})
        return pd.DataFrame(records)

    @staticmethod
    def _minmax(values: pd.Series) -> pd.Series:
        x = values.astype(float); lo, hi = float(x.min()), float(x.max())
        if hi - lo <= 1e-12:
            return pd.Series(np.ones(len(x)) if hi > 0 else np.zeros(len(x)), index=x.index)
        return (x - lo) / (hi - lo)

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k_frames: int = 200, top_k_videos: int = 100,
                 per_video_k: int = 5, beit3_query_embedding: np.ndarray | None = None,
                 scoring_query: str | None = None) -> pd.DataFrame:
        frames = self.frame_index.search_frames(query_embedding, top_k=top_k_frames)
        if not frames:
            return pd.DataFrame()
        rows = pd.DataFrame([x.__dict__ for x in frames])
        candidates = self._aggregate_frame_evidence(rows, per_video_k=per_video_k)
        if beit3_query_embedding is not None:
            if self.beit3_index is None:
                raise ValueError("BEiT-3 query embedding supplied but no BEiT-3 index is configured")
            beit3_candidates = self._aggregate_beit3(self.beit3_index.search(beit3_query_embedding, top_k=top_k_frames))
            candidates = candidates.merge(beit3_candidates, on="video_id", how="outer")
            candidates["clip_norm"] = self._minmax(candidates["retrieval_score"].fillna(0.0))
            candidates["beit3_norm"] = self._minmax(candidates["beit3_score"].fillna(0.0))
            w = self.beit3_weight
            candidates["retrieval_score"] = (1.0 - w) * candidates["clip_norm"] + w * candidates["beit3_norm"]
            candidates["score"] = candidates["retrieval_score"]
            candidates = candidates.sort_values("retrieval_score", ascending=False).reset_index(drop=True)
            candidates["best_frame_idx"] = candidates["best_frame_idx"].fillna(0).astype(int)
            candidates["best_frame_id"] = candidates["best_frame_id"].fillna(candidates["beit3_best_frame_id"]).astype(int)
            candidates["best_pts_time"] = candidates["best_pts_time"].fillna(candidates["beit3_pts_time"])
            candidates["object_path"] = candidates.get("object_path", pd.Series([""] * len(candidates))).fillna("")
            for col in ("retrieval_best_score", "retrieval_topk_mean", "retrieval_score_std"):
                candidates[col] = candidates.get(col, pd.Series([0.0] * len(candidates))).fillna(0.0)
        candidates = self._enrich_support_paths(candidates)
        score_query = scoring_query or query
        try:
            fused = self.reranker.score_manifest(score_query, candidates)
        except (AttributeError, TypeError, KeyError, ValueError):
            fused = candidates.copy(); fused["fused_score"] = fused["retrieval_score"]
        fused["multimodal_score"] = fused["fused_score"]
        keep = ["video_id", "retrieval_score", "multimodal_score", "best_frame_idx", "best_frame_id", "best_pts_time",
                "object_path", "retrieval_best_score", "retrieval_topk_mean", "retrieval_score_std"]
        candidates = fused[keep].copy(); candidates["temporal_score"] = 0.0
        return rerank_candidates(candidates, self.ranking_weights).head(top_k_videos).reset_index(drop=True)

    def localize(self, candidate: pd.Series, frame_scorer: FrameScorerFn | FrameScorerProtocol | None = None,
                 radius_frames: int = 24, max_decode_frames: int = 96) -> LocalizedEvent:
        video_id = str(candidate["video_id"]); coarse = int(candidate["best_frame_id"])
        video_path = self.video_resolver.resolve(video_id)
        if video_path is None:
            raise FileNotFoundError(f"Source video not found for {video_id} under {self.videos_dir}")
        info = probe_video(video_path)
        if info.frame_count <= 0:
            raise RuntimeError(f"Video has no decodable frames: {video_path}")
        start = max(0, coarse - radius_frames); end = min(info.frame_count - 1, coarse + radius_frames)
        frame_ids = list(range(start, end + 1))
        if len(frame_ids) > max_decode_frames:
            positions = np.linspace(0, len(frame_ids) - 1, max_decode_frames, dtype=int)
            frame_ids = [frame_ids[int(i)] for i in positions]
        decoded = list(iter_frame_ids(video_path, frame_ids))
        if not decoded:
            raise RuntimeError(f"Unable to decode temporal window for {video_id}")
        observed_ids = [frame_id for frame_id, _ in decoded]
        if frame_scorer is None:
            semantic = coarse if coarse in observed_ids else min(observed_ids, key=lambda frame_id: abs(frame_id - coarse))
            score = float(candidate.get("multimodal_score", candidate.get("retrieval_score", 0.0)))
            return LocalizedEvent(video_id, coarse, start, end, semantic, score)
        frames = [frame for _, frame in decoded]; scores = [float(x) for x in frame_scorer(frames)]
        if len(scores) != len(observed_ids):
            raise ValueError("frame_scorer must return one score per decoded frame")
        semantic = select_peak_frame(observed_ids, scores); windows = merge_frame_hits(video_id, observed_ids, scores, max_gap=1)
        if windows:
            best_window = max(windows, key=lambda w: w.score)
            local_pairs = [(fid, s) for fid, s in zip(observed_ids, scores) if best_window.start_frame <= fid <= best_window.end_frame]
            local_ids = [x[0] for x in local_pairs]; local_scores = [x[1] for x in local_pairs]
            return LocalizedEvent(video_id, coarse, best_window.start_frame, best_window.end_frame, semantic,
                                  temporal_event_score(local_ids, local_scores, self.temporal_config))
        return LocalizedEvent(video_id, coarse, start, end, semantic, temporal_event_score(observed_ids, scores, self.temporal_config))

    def run(self, query: str, query_embedding: np.ndarray, top_k: int = 100,
            frame_scorer: FrameScorerFn | FrameScorerProtocol | None = None, radius_frames: int = 24,
            max_decode_frames: int = 96, beit3_query_embedding: np.ndarray | None = None,
            scoring_query: str | None = None) -> pd.DataFrame:
        top_k_frames = max(top_k * 10, 1000); top_k_videos = min(max(top_k, 1), 100)
        candidates = self.retrieve(query, query_embedding, top_k_frames=top_k_frames, top_k_videos=top_k_videos,
                                    per_video_k=5, beit3_query_embedding=beit3_query_embedding, scoring_query=scoring_query)
        if candidates.empty:
            return candidates
        events: list[dict[str, object]] = []
        for _, candidate in candidates.iterrows():
            event = self.localize(candidate, frame_scorer=frame_scorer, radius_frames=radius_frames, max_decode_frames=max_decode_frames)
            events.append({**candidate.to_dict(), "temporal_start_frame": event.start_frame, "temporal_end_frame": event.end_frame,
                           "semantic_keyframe": event.semantic_keyframe, "temporal_score": event.score})
        return rerank_candidates(pd.DataFrame(events), self.ranking_weights).head(top_k).reset_index(drop=True)

    def write_candidates(self, candidates: pd.DataFrame, output: str | Path, top_k: int = 100) -> None:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        top_k_submission(candidates, top_k).to_json(path, orient="records", force_ascii=False, indent=2)


AICPipeline = RetrievalPipeline
