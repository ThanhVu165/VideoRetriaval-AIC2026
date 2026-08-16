from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class VQARequest:
    query_id: str
    question: str
    video_id: str
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class VQAResult:
    query_id: str
    video_id: str
    frame_ids: tuple[int, ...]
    answer: str
    confidence: float | None = None


class VLMAnswerer(Protocol):
    """Adapter contract for a local VLM/video-language model."""

    def answer(self, question: str, frames: Sequence[object]) -> str:
        ...


def build_vqa_prompt(question: str) -> str:
    return (
        "Answer the question using only the provided video frames. "
        "Be concise and do not invent details.\n\n"
        f"Question: {question}"
    )


def extract_answer(
    request: VQARequest,
    frames: Sequence[object],
    answerer: VLMAnswerer,
) -> VQAResult:
    if not frames:
        raise ValueError("VQA requires at least one decoded frame")
    if len(frames) != len(request.frame_ids):
        raise ValueError("number of frames must match request.frame_ids")
    answer = answerer.answer(build_vqa_prompt(request.question), frames).strip()
    return VQAResult(
        query_id=request.query_id,
        video_id=request.video_id,
        frame_ids=request.frame_ids,
        answer=answer,
    )
