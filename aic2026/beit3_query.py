from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache
def load_beit3(device: str):
    """Load the BEiT-3 Large COCO-retrieval text encoder."""
    try:
        from transformers import XLMRobertaTokenizer
    except ImportError as exc:
        raise ImportError("BEiT-3 query encoding requires transformers") from exc

    home = os.environ.get("AIC_BEIT3_HOME", "")
    checkpoint = os.environ.get("AIC_BEIT3_CHECKPOINT", "")
    spm = os.environ.get("AIC_BEIT3_SPM", "")
    if not all((home, checkpoint, spm)):
        raise RuntimeError(
            "Set AIC_BEIT3_HOME, AIC_BEIT3_CHECKPOINT and AIC_BEIT3_SPM."
        )
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"BEiT-3 checkpoint not found: {checkpoint}")
    if not Path(spm).is_file():
        raise FileNotFoundError(f"BEiT-3 SentencePiece model not found: {spm}")

    if home not in sys.path:
        sys.path.insert(0, home)
    import modeling_finetune
    import utils

    model = modeling_finetune.beit3_large_patch16_384_retrieval()
    utils.load_model_and_may_interpolate(checkpoint, model, "model|module", "")
    model = model.to(device).eval()
    tokenizer = XLMRobertaTokenizer(spm)
    return model, tokenizer


class BEiT3QueryEncoder:
    """Text encoder compatible with the 1024-D BEiT-3 retrieval index."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model, self._tokenizer = load_beit3(device)

    def encode_one(self, query: str) -> np.ndarray:
        import torch
        import faiss

        tokens = self._tokenizer(
            query,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        with torch.inference_mode():
            _, language = self._model(
                text_description=tokens,
                padding_mask=tokens.eq(self._tokenizer.pad_token_id),
                only_infer=True,
            )
        vector = language.cpu().numpy().astype("float32").reshape(-1)
        faiss.normalize_L2(vector.reshape(1, -1))
        return vector

    def encode_many(self, queries: list[str]) -> np.ndarray:
        return np.vstack([self.encode_one(query) for query in queries])
