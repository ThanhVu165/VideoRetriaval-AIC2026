from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .multimodal import FusionWeights, MultimodalReranker, load_json_text
from .ranking import RankingWeights, rerank_candidates, top_k_submission
from .retrieval import FrameIndex
from .temporal import TemporalWindow, merge_frame_hits, refine_window
from .video import iter_frame_ids, probe_video

FrameScorer = Callable[[Sequence[object]], Sequence[float]]


@dataclass(frozen=True)
class LocalizedEvent:
    video_id: str
    coarse_frame_id: int
    start_frame: int
    end_frame: int
    semantic_keyframe: int
    score: float


class AICPipeline:
    """Coarse-to-fine AIC retrieval pipeline.

    Query embedding generation and VLM inference are injected as adapters.
    This keeps the pipeline independent of an unverified competition query
    format while making the retrieval/localization stages executable today.
    """

    def __init__(
        self,
        frame_index: FrameIndex,
        videos_dir: str | Path,
        media_info_dir: str | Path | None = None,
        reranker: MultimodalReranker | None = None,
        ranking_weights: RankingWeights | None = None,
    ):
        self.frame_index = frame_index
        self.videos_dir = Path(videos_dir)
        self.media_info_dir = Path(media_info_dir) if media_info_dir else None
        self.reranker = reranker or MultimodalReranker(FusionWeights())
        self.ranking_weights = ranking_weights or RankingWeights()

    def _metadata_text(self, video_id: str) -> str:
        if self.media_info_dir is None:
            return ""
        path = self.media_info_dir / f"{video_id}.json"
        return load_json_text(path)

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k_frames: int = 200, top_k_videos: int = 100) -> pd.DataFrame:
        frames = self.frame_index.search_frames(query_embedding, top_k=top_k_frames)
        if not frames:
            return pd.DataFrame()
        rows = pd.DataFrame([x.__dict__ for x in frames])
        rows["retrieval_score"] = rows["score"].astype(float)
        rows["object_path"] = rows.apply(
            lambda r: str(self.frame_index.manifest.iloc[r.name].get("object_path", "")), axis=1
        )
        rows["metadata_text"] = rows["video_id"].map(self._metadata_text)

        fused = self.reranker.score_manifest(query, rows.rename(columns={"score": "score"}))
        fused["multimodal_score"] = fused["fused_score"]
        best = fused.sort_values("fused_score", ascending=False).drop_duplicates("video_id")
        best = best.rename(columns={"keyframe_idx": "best_frame_idx", "original_frame_id": "best_frame_id", "pts_time": "best_pts_time"})
        candidates = best[[
            "video_id", "retrieval_score", "multimodal_score", "best_frame_idx", "best_frame_id", "best_pts_time"
        ]].copy()
        candidates["temporal_score"] = 0.0
        return rerank_candidates(candidates, self.ranking_weights).head(top_k_videos).reset_index(drop=True)

    def localize(
        self,
        candidate: pd.Series,
        frame_scorer: FrameScorer | None = None,
        radius_frames: int = 15,
        max_decode_frames: int = 64,
    ) -> LocalizedEvent:
        video_id = str(candidate["video_id"])
        coarse = int(candidate["best_frame_id"])
        video_path = self.videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            alternatives = list(self.videos_dir.glob(f"{video_id}.*"))
            if not alternatives:
                raise FileNotFoundError(f"Source video not found for {video_id}")
            video_path = alternatives[0]
        info = probe_video(video_path)
        start = max(0, coarse - radius_frames)
        end = min(info.frame_count - 1, coarse + radius_frames)
        frame_ids = list(range(start, end + 1))
        if len(frame_ids) > max_decode_frames:
            step = max(1, len(frame_ids) // max_decode_frames)
            frame_ids = frame_ids[::step]
            if frame_ids[-1] != end:
                frame_ids.append(end)
        decoded = list(iter_frame_ids(video_path, frame_ids))
        if not decoded:
            raise RuntimeError(f"Unable to decode temporal window for {video_id}")

        observed_ids = [frame_id for frame_id, _ in decoded]
        if frame_scorer is None:
            semantic = coarse if coarse in observed_ids else observed_ids[len(observed_ids) // 2]
            score = float(candidate.get("multimodal_score", candidate.get("retrieval_score", 0.0)))
            return LocalizedEvent(video_id, coarse, start, end, semantic, score)

        frames = [frame for _, frame in decoded]
        scores = [float(x) for x in frame_scorer(frames)]
        if len(scores) != len(observed_ids):
            raise ValueError("frame_scorer must return one score per decoded frame")
        semantic = refine_window(start, end, dict(zip(observed_ids, scores)))
        windows = merge_frame_hits(video_id, observed_ids, scores, max_gap=1)
        best_window = windows[0] if windows else TemporalWindow(video_id, start, end, max(scores))
        return LocalizedEvent(video_id, best_window.start_frame, best_window.start_frame, best_window.end_frame, semantic, best_window.score)

    def run(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 100,
        frame_scorer: FrameScorer | None = None,
        radius_frames: int = 15,
    ) -> pd.DataFrame:
        candidates = self.retrieve(query, query_embedding, top_k_frames=max(top_k * 2, 200), top_k_videos=top_k)
        if candidates.empty:
            return candidates
        events: list[dict[str, object]] = []
        for _, candidate in candidates.iterrows():
            event = self.localize(candidate, frame_scorer=frame_scorer, radius_frames=radius_frames)
            events.append({
                **candidate.to_dict(),
                "temporal_start_frame": event.start_frame,
                "temporal_end_frame": event.end_frame,
                "semantic_keyframe": event.semantic_keyframe,
                "temporal_score": event.score,
            })
        result = pd.DataFrame(events)
        return rerank_candidates(result, self.ranking_weights).head(top_k).reset_index(drop=True)

    def write_candidates(self, candidates: pd.DataFrame, output: str | Path, top_k: int = 100) -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        top_k_submission(candidates, top_k).to_json(path, orient="records", force_ascii=False, indent=2)
