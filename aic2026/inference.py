from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .clip_runtime import OpenCLIPRuntime
from .pipeline import AICPipeline


@dataclass(frozen=True)
class InferenceConfig:
    top_k_videos: int = 100
    retrieval_frames: int = 500
    temporal_radius: int = 24
    max_decode_frames: int = 96
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cpu"
    batch_size: int = 32


def build_runtime(config: InferenceConfig) -> OpenCLIPRuntime:
    return OpenCLIPRuntime(
        model_name=config.model_name,
        pretrained=config.pretrained,
        device=config.device,
        batch_size=config.batch_size,
    )


def retrieve_query(
    pipeline: AICPipeline,
    runtime: OpenCLIPRuntime,
    query: str,
    config: InferenceConfig | None = None,
):
    cfg = config or InferenceConfig()
    query_embedding = runtime.encode_text([query])[0]

    def scorer(frames: Sequence[np.ndarray]) -> Sequence[float]:
        return runtime.score_frames(frames, query_embedding)

    return pipeline.run(
        query,
        query_embedding,
        top_k=cfg.top_k_videos,
        frame_scorer=scorer,
        radius_frames=cfg.temporal_radius,
    )


def load_pipeline(
    manifest: str | Path,
    clip_features: str | Path,
    videos_dir: str | Path,
    media_info_dir: str | Path | None = None,
) -> AICPipeline:
    from .retrieval import FrameIndex

    index = FrameIndex.from_files(manifest, clip_features)
    return AICPipeline(index, videos_dir=videos_dir, media_info_dir=media_info_dir)
