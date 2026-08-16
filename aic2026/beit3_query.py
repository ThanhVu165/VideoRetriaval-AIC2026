from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


_BEIT3_CHECKPOINT_NAME = "beit3_large_patch16_384_coco_retrieval.pth"
_BEIT3_SPM_NAME = "beit3.spm"


def _default_beit3_home() -> Path:
    # Keep large model assets outside the Git repository history while making
    # the standard local layout work without manual environment variables.
    return Path(__file__).resolve().parent.parent / "models" / "beit3"


@lru_cache
def load_beit3(device: str):
    """Load the BEiT-3 Large COCO-retrieval text encoder."""
    try:
        from transformers import XLMRobertaTokenizer
    except ImportError as exc:
        raise ImportError("BEiT-3 query encoding requires transformers") from exc

    default_home = _default_beit3_home()
    home = Path(os.environ.get("AIC_BEIT3_HOME", str(default_home)))
    checkpoint = Path(
        os.environ.get(
            "AIC_BEIT3_CHECKPOINT",
            str(home / _BEIT3_CHECKPOINT_NAME),
        )
    )
    spm = Path(
        os.environ.get(
            "AIC_BEIT3_SPM",
            str(home / _BEIT3_SPM_NAME),
        )
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "BEiT-3 retrieval checkpoint not found. Expected: "
            f"{checkpoint}. Download {_BEIT3_CHECKPOINT_NAME} and place it "
            "under models/beit3/, or set AIC_BEIT3_CHECKPOINT."
        )
    if not spm.is_file():
        raise FileNotFoundError(
            "BEiT-3 SentencePiece model not found. Expected: "
            f"{spm}. Download beit3.spm and place it under models/beit3/, "
            "or set AIC_BEIT3_SPM."
        )
    if not home.is_dir():
        raise FileNotFoundError(f"BEiT-3 source directory not found: {home}")

    home_str = str(home.resolve())
    if home_str not in sys.path:
        sys.path.insert(0, home_str)
    import modeling_finetune
    import utils

    model = modeling_finetune.beit3_large_patch16_384_retrieval()
    utils.load_model_and_may_interpolate(str(checkpoint), model, "model|module", "")
    model = model.to(device).eval()
    tokenizer = XLMRobertaTokenizer(str(spm))
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
