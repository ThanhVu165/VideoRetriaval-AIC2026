from __future__ import annotations

from pathlib import Path

from .pipeline import RetrievalPipeline
from .temporal_grounding import CLIPTemporalGrounder
from .vqa import VQARequest, extract_answer
from .video import iter_frame_ids


class VQARunner:
    """Retrieve a video, localize a relevant window, and call a VLM."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        grounder: CLIPTemporalGrounder,
        answerer,
    ) -> None:
        self.pipeline = pipeline
        self.grounder = grounder
        self.answerer = answerer

    def run(
        self,
        query_id: str,
        question: str,
        top_k_videos: int = 10,
        radius_frames: int = 24,
        max_decode_frames: int = 16,
    ):
        embedding = self.grounder.text_embedding(question)
        candidates = self.pipeline.retrieve(
            question,
            embedding,
            top_k_frames=max(top_k_videos * 10, 200),
            top_k_videos=top_k_videos,
            per_video_k=5,
        )
        if candidates.empty:
            raise RuntimeError("VQA retrieval returned no candidate videos")

        candidate = candidates.iloc[0]
        localized = self.pipeline.localize(
            candidate,
            frame_scorer=self.grounder.scorer(embedding),
            radius_frames=radius_frames,
            max_decode_frames=max_decode_frames * 4,
        )
        video_path = self.pipeline.video_resolver.resolve(localized.video_id)
        if video_path is None:
            raise FileNotFoundError(f"Source video not found for {localized.video_id}")

        start = localized.start_frame
        end = localized.end_frame
        ids = list(range(start, end + 1))
        if len(ids) > max_decode_frames:
            import numpy as np

            positions = np.linspace(0, len(ids) - 1, max_decode_frames, dtype=int)
            ids = [ids[int(i)] for i in positions]
        decoded = list(iter_frame_ids(video_path, ids))
        if not decoded:
            raise RuntimeError(f"Unable to decode VQA frames for {localized.video_id}")

        request = VQARequest(
            query_id=str(query_id),
            question=question,
            video_id=localized.video_id,
            frame_ids=tuple(frame_id for frame_id, _ in decoded),
        )
        return extract_answer(
            request,
            [frame for _, frame in decoded],
            self.answerer,
        )
