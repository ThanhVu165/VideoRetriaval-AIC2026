from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class TransformersVLMAnswerer:
    """Optional local Hugging Face VLM adapter.

    The backend is deliberately optional because model size and GPU
    requirements vary. A vision-language model that supports the
    ``image-text-to-text`` pipeline can be supplied at runtime.
    """

    model_id: str
    device: str = "auto"
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "TransformersVLMAnswerer requires transformers."
            ) from exc
        kwargs = {"model": self.model_id}
        if self.device != "auto":
            kwargs["device"] = self.device
        self._pipeline = pipeline("image-text-to-text", **kwargs)

    @staticmethod
    def _normalize_output(output: object) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, list) and output:
            return TransformersVLMAnswerer._normalize_output(output[-1])
        if isinstance(output, dict):
            for key in ("generated_text", "text", "answer"):
                if key in output:
                    return str(output[key])
        return str(output)

    def answer(self, question: str, frames: Sequence[object]) -> str:
        if not frames:
            raise ValueError("VQA requires at least one frame")

        # The pipeline API differs slightly across VLM families. Try the
        # message form first, then the simpler text/images form.
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        try:
            output = self._pipeline(
                text=messages,
                max_new_tokens=self.max_new_tokens,
            )
            return self._normalize_output(output).strip()
        except (TypeError, ValueError, KeyError):
            output = self._pipeline(
                text=question,
                images=list(frames),
                max_new_tokens=self.max_new_tokens,
            )
            return self._normalize_output(output).strip()
