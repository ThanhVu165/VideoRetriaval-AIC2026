from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class OpenCLIPRuntime:
    """Optional OpenCLIP runtime for query encoding and fine frame scoring.

    Keep the checkpoint configurable. To reproduce the supplied BTC CLIP
    features, use the exact compatible ViT-B/32 checkpoint used to generate
    those features rather than silently selecting another pretrained model.
    """

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cpu"
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "OpenCLIP runtime requires torch and open_clip_torch; "
                "install them separately from the lightweight core requirements."
            ) from exc
        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(texts)).to(self.device)
        with self._torch.inference_mode():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32)

    def score_frames(self, frames: Sequence[np.ndarray], text_embedding: np.ndarray) -> list[float]:
        if not frames:
            return []
        from PIL import Image
        import cv2

        text = np.asarray(text_embedding, dtype=np.float32).reshape(1, -1)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        scores: list[float] = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            images = []
            for frame in batch:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(self._preprocess(Image.fromarray(rgb)))
            tensor = self._torch.stack(images).to(self.device)
            with self._torch.inference_mode():
                image_features = self._model.encode_image(tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                batch_scores = image_features @ self._torch.from_numpy(text.T).to(self.device)
            scores.extend(batch_scores.squeeze(-1).detach().cpu().numpy().astype(float).tolist())
        return scores
