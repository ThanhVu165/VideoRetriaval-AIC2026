from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .clip_runtime import OpenCLIPRuntime


@dataclass
class CLIPTemporalGrounder:
    """Fine-grained frame scorer using the same CLIP family as retrieval.

    This is a baseline temporal grounder, not a learned temporal model. It
    exists so the pipeline can actually rescore decoded original frames
    instead of falling back to the coarse keyframe returned by FAISS.
    """

    runtime: OpenCLIPRuntime

    @classmethod
    def create(
        cls,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> "CLIPTemporalGrounder":
        return cls(OpenCLIPRuntime(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            batch_size=batch_size,
        ))

    def text_embedding(self, query: str) -> np.ndarray:
        return self.runtime.encode_text([query])[0]

    def score(self, frames: Sequence[np.ndarray], query_embedding: np.ndarray) -> list[float]:
        return self.runtime.score_frames(frames, query_embedding)

    def scorer(self, query_embedding: np.ndarray):
        embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        def _score(frames: Sequence[np.ndarray]) -> list[float]:
            return self.score(frames, embedding)

        return _score
