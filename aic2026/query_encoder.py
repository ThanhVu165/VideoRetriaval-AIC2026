from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CLIPQueryEncoder:
    """Encode text queries into the 512-D CLIP space used by the dataset index.

    The checkpoint is explicit on purpose: the text encoder must be compatible
    with the supplied BTC ViT-B/32 visual features. Do not silently substitute
    another CLIP checkpoint.
    """

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Text query encoding requires torch and open_clip_torch. "
                "Install them with: pip install torch torchvision open_clip_torch"
            ) from exc
        self._torch = torch
        self._open_clip = open_clip
        self._model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode(self, queries: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(queries)).to(self.device)
        with self._torch.inference_mode():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        result = features.detach().cpu().numpy().astype(np.float32)
        if result.ndim != 2 or result.shape[1] != 512:
            raise ValueError(f"expected 512-D query embeddings, got {result.shape}")
        return result

    def encode_one(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
