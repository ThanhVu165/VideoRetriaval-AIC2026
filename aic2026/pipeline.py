from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import LocalizedEvent, merge_frame_hits, select_peak_frame, temporal_event_score
from .fine_scoring import FrameScorer
from .inference import build_query_embedding
from .multimodal import MultiModalReranker
from .ranking import rerank_candidates, top_k_submission
from .retrieval import FrameIndex
from .temporal import TemporalConfig
from .video import iter_frame_ids, probe_video
from .vqa import answer_query


class VideoResolver:
    def __init__(self, videos_dir: str | Path):
        self.videos_dir = Path(videos_dir)
        self._paths = {
            p.stem: p
            for p in self.videos_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}
        }

    @property
    def count(self) -> int:
        return len(self._paths)

    def resolve(self, video_id: str) -> Path | None:
        return self._paths.get(video_id)


class RetrievalPipeline:
    def __init__(self, frame_index: FrameIndex, videos_dir: str | Path, ranking_weights: dict[str, float] | None = None, temporal_config: TemporalConfig | None = None):
        self.frame_index = frame_index
        self.videos_dir = Path(videos_dir)
        self.video_resolver = VideoResolver(videos_dir)
        self.reranker = MultiModalReranker(frame_index.manifest)
        self.ranking_weights = ranking_weights or {"retrieval_score": 0.55, "multimodal_score": 0.30, "temporal_score": 0.15}
        self.temporal_config = temporal_config or TemporalConfig()

    def _metadata_text(self, video_id: str) -> str:
        return ""

    def _aggregate_frame_evidence(self, rows: pd.DataFrame, per_video_k: int = 5) -> pd.DataFrame:
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

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k_frames: int = 200, top_k_videos: int = 100, per_video_k: int = 5) -> pd.DataFrame:
        frames = self.frame_index.search_frames(query_embedding, top_k=top_k_frames)
        if not frames:
            return pd.DataFrame()
        rows = pd.DataFrame([x.__dict__ for x in frames])
        candidates = self._aggregate_frame_evidence(rows, per_video_k=per_video_k)
        candidates["metadata_text"] = candidates["video_id"].map(self._metadata_text)
        fused = self.reranker.score_manifest(query, candidates)
        fused["multimodal_score"] = fused["fused_score"]
        keep = ["video_id", "retrieval_score", "multimodal_score", "best_frame_idx", "best_frame_id", "best_pts_time", "object_path", "retrieval_best_score", "retrieval_topk_mean", "retrieval_score_std"]
        candidates = fused[keep].copy()
        candidates["temporal_score"] = 0.0
        return rerank_candidates(candidates, self.ranking_weights).head(top_k_videos).reset_index(drop=True)

    def localize(self, candidate: pd.Series, frame_scorer: FrameScorer | None = None, radius_frames: int = 24, max_decode_frames: int = 96) -> LocalizedEvent:
        video_id = str(candidate["video_id"])
        coarse = int(candidate["best_frame_id"])
        video_path = self.video_resolver.resolve(video_id)
        if video_path is None:
            raise FileNotFoundError(
                f"Source video not found for {video_id} under {self.videos_dir} "
                f"(discovered {self.video_resolver.count} videos recursively)"
            )

        info = probe_video(video_path)
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
            semantic = coarse if coarse in observed_ids else observed_ids[len(observed_ids) // 2]
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
            local_ids = [f for f in observed_ids if best_window.start_frame <= f <= best_window.end_frame]
            local_scores = [scores[observed_ids.index(f)] for f in local_ids]
            event_score = temporal_event_score(local_ids, local_scores, self.temporal_config)
            return LocalizedEvent(video_id, coarse, best_window.start_frame, best_window.end_frame, semantic, event_score)

        return LocalizedEvent(video_id, coarse, start, end, semantic, temporal_event_score(observed_ids, scores, self.temporal_config))

    def run(self, query: str, query_embedding: np.ndarray, top_k: int = 100, frame_scorer: FrameScorer | None = None, radius_frames: int = 24, max_decode_frames: int = 96) -> pd.DataFrame:
        # Retrieval needs a broad frame pool, but temporal decoding should only
        # inspect the requested candidate-video count. In particular, a small
        # debug top-k must not silently expand to 20 temporal decodes.
        top_k_frames = max(top_k * 10, 1000)
        top_k_videos = min(max(top_k, 10), 100)
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
        top_k_submission(candidates, top_k).to_json(path, orient="records", force_ascii=False, indent=2)
