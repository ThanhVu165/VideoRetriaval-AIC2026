from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .clip_runtime import OpenCLIPRuntime
from .evidence import EvidenceStore
from .pipeline import AICPipeline


@dataclass(frozen=True)
class InferenceConfig:
    top_k_videos: int = 100
    retrieval_frames: int = 1000
    temporal_radius: int = 24
    max_decode_frames: int = 96
    model_name: str = "ViT-B-32-quickgelu"
    pretrained: str = "openai"
    device: str = "cpu"
    batch_size: int = 32
    evidence_rrf_k: int = 60


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
    translated_query: str | None = None,
):
    """Use the canonical retrieval/temporal pipeline from one inference API."""
    cfg = config or InferenceConfig()
    query_embedding = runtime.encode_text([query])[0]
    scoring_query = query
    if translated_query and translated_query.strip():
        translated_embedding = runtime.encode_text([translated_query])[0]
        query_embedding = 0.20 * query_embedding + 0.80 * translated_embedding
        query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
        scoring_query = translated_query

    def scorer(frames: Sequence[np.ndarray]) -> Sequence[float]:
        return runtime.score_frames(frames, query_embedding)

    return pipeline.run(
        query,
        query_embedding,
        top_k=min(max(cfg.top_k_videos, 1), 100),
        frame_scorer=scorer,
        radius_frames=cfg.temporal_radius,
        max_decode_frames=cfg.max_decode_frames,
        scoring_query=scoring_query,
    )


def load_pipeline(
    manifest: str | Path,
    clip_features: str | Path,
    videos_dir: str | Path,
    media_info_dir: str | Path | None = None,
    objects_dir: str | Path | None = None,
    evidence_stores: Sequence[EvidenceStore] | None = None,
    evidence_rrf_k: int = 60,
) -> AICPipeline:
    from .retrieval import FrameIndex

    index = FrameIndex.from_files(manifest, clip_features)
    return AICPipeline(
        index,
        videos_dir=videos_dir,
        media_info_dir=media_info_dir,
        objects_dir=objects_dir,
        evidence_stores=evidence_stores,
        evidence_rrf_k=evidence_rrf_k,
    )
