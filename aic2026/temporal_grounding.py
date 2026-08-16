from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .clip_runtime import OpenCLIPRuntime


@dataclass
class CLIPTemporalGrounder:
    """Fine-grained CLIP scoring of decoded original-video frames.

    This reuses the same OpenAI CLIP ViT-B/32 family as the supplied BTC
    frame features. The caller supplies the final query embedding so query
    translation/fusion is performed exactly once in the online pipeline.
    """

    runtime: OpenCLIPRuntime

    @classmethod
    def create(
        cls,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> "CLIPTemporalGrounder":
        return cls(
            OpenCLIPRuntime(
                model_name=model_name,
                pretrained=pretrained,
                device=device,
                batch_size=batch_size,
            )
        )

    def scorer(self, text_embedding: np.ndarray):
        embedding = np.asarray(text_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            raise ValueError("text_embedding must be non-zero")
        embedding = embedding / norm

        def score(frames: Sequence[np.ndarray]) -> list[float]:
            return self.runtime.score_frames(frames, embedding)

        return score
