from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .fine_scoring import (
    FrameScorerProtocol,
    TemporalScoreConfig,
    select_peak_frame,
    temporal_event_score,
)
from .multimodal import MultimodalReranker
from .ranking import RankingWeights, rerank_candidates, top_k_submission
from .retrieval import FrameIndex
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
    """Resolve video IDs recursively under the configured video directory."""

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

    def resolve(self, video_id: str) -> Path | None:
        return self._paths.get(video_id)


class RetrievalPipeline:
    def __init__(
        self,
        frame_index: FrameIndex,
        videos_dir: str | Path,
        media_info_dir: str | Path | None = None,
        ranking_weights: RankingWeights | None = None,
        temporal_config: TemporalScoreConfig | None = None,
    ):
        self.frame_index = frame_index
        self.videos_dir = Path(videos_dir)
        self.media_info_dir = Path(media_info_dir) if media_info_dir else None
        self.video_resolver = VideoResolver(self.videos_dir)
        self.reranker = MultimodalReranker()
        self.ranking_weights = ranking_weights or RankingWeights(
            retrieval=0.55,
            multimodal=0.30,
            temporal=0.15,
        )
        self.temporal_config = temporal_config or TemporalScoreConfig()

    def _metadata_text(self, video_id: str) -> str:
        """Load optional per-video metadata as text without making it mandatory."""
        if self.media_info_dir is None:
            return ""
        candidates = [
            self.media_info_dir / f"{video_id}.json",
            self.media_info_dir / video_id / "media_info.json",
            self.media_info_dir / video_id / f"{video_id}.json",
        ]
        for path in candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.dumps(json.load(f), ensure_ascii=False)
            except (OSError, ValueError, TypeError):
                continue
        return ""

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
            records.append(
                {
                    "video_id": str(video_id),
                    "score": 0.75 * best_score + 0.25 * top_mean,
                    "retrieval_score": 0.75 * best_score + 0.25 * top_mean,
                    "best_frame_idx": int(best["keyframe_idx"]),
                    "best_frame_id": int(best["original_frame_id"]),
                    "best_pts_time": float(best["pts_time"]),
                    "object_path": str(best.get("object_path", "")),
                    "retrieval_best_score": best_score,
                    "retrieval_topk_mean": top_mean,
                    "retrieval_score_std": float(np.std(scores)) if len(scores) > 1 else 0.0,
                }
            )
        return pd.DataFrame(records).sort_values("retrieval_score", ascending=False).reset_index(drop=True)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k_frames: int = 200,
        top_k_videos: int = 100,
        per_video_k: int = 5,
    ) -> pd.DataFrame:
        frames = self.frame_index.search_frames(query_embedding, top_k=top_k_frames)
        if not frames:
            return pd.DataFrame()
        rows = pd.DataFrame([x.__dict__ for x in frames])
        candidates = self._aggregate_frame_evidence(rows, per_video_k=per_video_k)
        candidates["metadata_text"] = [self._metadata_text(v) for v in candidates["video_id"]]
        try:
            fused = self.reranker.score_manifest(query, candidates)
        except (AttributeError, TypeError, KeyError, ValueError):
            fused = candidates.copy()
            fused["fused_score"] = fused["retrieval_score"]

        fused["multimodal_score"] = fused["fused_score"]
        keep = [
            "video_id",
            "retrieval_score",
            "multimodal_score",
            "best_frame_idx",
            "best_frame_id",
            "best_pts_time",
            "object_path",
            "retrieval_best_score",
            "retrieval_topk_mean",
            "retrieval_score_std",
        ]
        candidates = fused[keep].copy()
        candidates["temporal_score"] = 0.0
        return rerank_candidates(candidates, self.ranking_weights).head(top_k_videos).reset_index(drop=True)

    def localize(
        self,
        candidate: pd.Series,
        frame_scorer: FrameScorerFn | FrameScorerProtocol | None = None,
        radius_frames: int = 24,
        max_decode_frames: int = 96,
    ) -> LocalizedEvent:
        video_id = str(candidate["video_id"])
        coarse = int(candidate["best_frame_id"])
        video_path = self.video_resolver.resolve(video_id)
        if video_path is None:
            raise FileNotFoundError(
                f"Source video not found for {video_id} under {self.videos_dir} "
                f"(discovered {self.video_resolver.count} videos recursively)"
            )

        info = probe_video(video_path)
        if info.frame_count <= 0:
            raise RuntimeError(f"Video has no decodable frames: {video_path}")

        start = max(0, coarse - radius_frames)
        end = min(info.frame_count - 1, coarse + radius_frames)
        frame_ids = list(range(start, end + 1))
        if len(frame_ids) > max_decode_frames:
            positions = np.linspace(0, len(frame_ids) - 1, max_decode_frames, dtype=int)
            frame_ids = [frame_ids[int(i)] for i in positions]

        decoded = list(iter_frame_ids(video_path, frame_ids))
        if not decoded:
            raise RuntimeError(f"Unable to decode temporal window for {video_id}")
        observed_ids = [frame_id for frame_id, _ in decoded]

        if frame_scorer is None:
            semantic = coarse if coarse in observed_ids else min(
                observed_ids,
                key=lambda frame_id: abs(frame_id - coarse),
            )
            score = float(candidate.get("multimodal_score", candidate.get("retrieval_score", 0.0)))
            return LocalizedEvent(video_id, coarse, start, end, semantic, score)

        frames = [frame for _, frame in decoded]
        scores = [float(x) for x in frame_scorer(frames)]
        if len(scores) != len(observed_ids):
            raise ValueError("frame_scorer must return one score per decoded frame")

        semantic = select_peak_frame(observed_ids, scores)
        windows = merge_frame_hits(video_id, observed_ids, scores, max_gap=1)
        if windows:
            best_window = max(windows, key=lambda w: w.score)
            local_pairs = [
                (frame_id, score)
                for frame_id, score in zip(observed_ids, scores)
                if best_window.start_frame <= frame_id <= best_window.end_frame
            ]
            local_ids = [x[0] for x in local_pairs]
            local_scores = [x[1] for x in local_pairs]
            event_score = temporal_event_score(local_ids, local_scores, self.temporal_config)
            return LocalizedEvent(
                video_id,
                coarse,
                best_window.start_frame,
                best_window.end_frame,
                semantic,
                event_score,
            )

        return LocalizedEvent(
            video_id,
            coarse,
            start,
            end,
            semantic,
            temporal_event_score(observed_ids, scores, self.temporal_config),
        )

    def run(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 100,
        frame_scorer: FrameScorerFn | FrameScorerProtocol | None = None,
        radius_frames: int = 24,
        max_decode_frames: int = 96,
    ) -> pd.DataFrame:
        top_k_frames = max(top_k * 10, 1000)
        top_k_videos = min(max(top_k, 1), 100)
        candidates = self.retrieve(
            query,
            query_embedding,
            top_k_frames=top_k_frames,
            top_k_videos=top_k_videos,
            per_video_k=5,
        )
        if candidates.empty:
            return candidates

        events: list[dict[str, object]] = []
        for _, candidate in candidates.iterrows():
            event = self.localize(
                candidate,
                frame_scorer=frame_scorer,
                radius_frames=radius_frames,
                max_decode_frames=max_decode_frames,
            )
            events.append(
                {
                    **candidate.to_dict(),
                    "temporal_start_frame": event.start_frame,
                    "temporal_end_frame": event.end_frame,
                    "semantic_keyframe": event.semantic_keyframe,
                    "temporal_score": event.score,
                }
            )

        result = pd.DataFrame(events)
        return rerank_candidates(result, self.ranking_weights).head(top_k).reset_index(drop=True)

    def write_candidates(self, candidates: pd.DataFrame, output: str | Path, top_k: int = 100) -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        top_k_submission(candidates, top_k).to_json(
            path,
            orient="records",
            force_ascii=False,
            indent=2,
        )


AICPipeline = RetrievalPipeline
