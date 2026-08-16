from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .alignment import SemanticKeyframe, monotonic_event_alignment
from .pipeline import RetrievalPipeline
from .temporal_grounding import CLIPTemporalGrounder
from .video import iter_frame_ids, probe_video


@dataclass(frozen=True)
class TRAKEResult:
    video_id: str
    events: tuple[SemanticKeyframe, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "events": [asdict(event) for event in self.events],
        }


class TRAKEEngine:
    """End-to-end baseline for ordered semantic-event keyframe retrieval.

    Stage 1 retrieves a small video candidate set for every event. Stage 2
    chooses the most consistent video. Stage 3 decodes original frames once,
    scores every event against those frames, and applies monotonic dynamic
    programming so the returned keyframes preserve event order.
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        grounder: CLIPTemporalGrounder,
        temporal_margin: int = 32,
        max_decode_frames: int = 512,
    ) -> None:
        if temporal_margin < 0 or max_decode_frames <= 0:
            raise ValueError("temporal_margin must be >= 0 and max_decode_frames > 0")
        self.pipeline = pipeline
        self.grounder = grounder
        self.temporal_margin = temporal_margin
        self.max_decode_frames = max_decode_frames

    def _select_video(
        self,
        event_queries: Sequence[str],
        event_embeddings: Sequence[np.ndarray],
        top_k_videos: int,
    ) -> tuple[str, list[pd.DataFrame]]:
        per_event: list[pd.DataFrame] = []
        votes: dict[str, float] = {}
        for query, embedding in zip(event_queries, event_embeddings):
            candidates = self.pipeline.retrieve(
                query,
                embedding,
                top_k_frames=max(top_k_videos * 10, 200),
                top_k_videos=top_k_videos,
                per_video_k=5,
            )
            per_event.append(candidates)
            for rank, (_, row) in enumerate(candidates.iterrows()):
                video_id = str(row["video_id"])
                score = float(row.get("rank_score", row.get("retrieval_score", 0.0)))
                votes[video_id] = votes.get(video_id, 0.0) + score + 1.0 / (rank + 1)
        if not votes:
            raise RuntimeError("TRAKE retrieval returned no candidate videos")
        video_id = max(votes.items(), key=lambda item: item[1])[0]
        return video_id, per_event

    @staticmethod
    def _coarse_frames(video_id: str, per_event: Sequence[pd.DataFrame]) -> list[int]:
        frames: list[int] = []
        for candidates in per_event:
            if candidates.empty:
                continue
            rows = candidates[candidates["video_id"].astype(str) == video_id]
            if rows.empty:
                continue
            frames.append(int(rows.iloc[0]["best_frame_id"]))
        return frames

    def _decode_frame_ids(self, video_path: Path, coarse_frames: Sequence[int]) -> list[int]:
        info = probe_video(video_path)
        if info.frame_count <= 0:
            raise RuntimeError(f"Video has no decodable frames: {video_path}")
        if not coarse_frames:
            raise RuntimeError("No coarse event frames were found for the selected video")
        start = max(0, min(coarse_frames) - self.temporal_margin)
        end = min(info.frame_count - 1, max(coarse_frames) + self.temporal_margin)
        ids = list(range(start, end + 1))
        if len(ids) <= self.max_decode_frames:
            return ids
        positions = np.linspace(0, len(ids) - 1, self.max_decode_frames, dtype=int)
        return [ids[int(i)] for i in positions]

    def run(
        self,
        event_queries: Sequence[str],
        top_k_videos: int = 50,
        min_separation: int = 0,
    ) -> TRAKEResult:
        queries = [str(q).strip() for q in event_queries if str(q).strip()]
        if not queries:
            raise ValueError("TRAKE requires at least one non-empty event query")

        embeddings = [self.grounder.text_embedding(query) for query in queries]
        video_id, per_event = self._select_video(queries, embeddings, top_k_videos)
        video_path = self.pipeline.video_resolver.resolve(video_id)
        if video_path is None:
            raise FileNotFoundError(f"Source video not found for {video_id}")

        coarse_frames = self._coarse_frames(video_id, per_event)
        if len(coarse_frames) != len(queries):
            raise RuntimeError(
                "Selected video is missing one or more event retrievals; "
                "increase top_k_videos or use a stronger event retriever."
            )

        frame_ids = self._decode_frame_ids(video_path, coarse_frames)
        decoded = list(iter_frame_ids(video_path, frame_ids))
        if not decoded:
            raise RuntimeError(f"Unable to decode TRAKE temporal span for {video_id}")
        observed_ids = [frame_id for frame_id, _ in decoded]
        frames = [frame for _, frame in decoded]

        score_matrix = np.asarray(
            [self.grounder.score(frames, embedding) for embedding in embeddings],
            dtype=np.float32,
        )
        events = monotonic_event_alignment(
            [f"event_{idx + 1}" for idx in range(len(queries))],
            observed_ids,
            score_matrix,
            min_separation=min_separation,
        )
        return TRAKEResult(video_id=video_id, events=tuple(events))
