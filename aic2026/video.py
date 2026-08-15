from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int

    @property
    def duration_frames(self) -> int:
        return self.frame_count


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS for {path}: {fps}")
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoInfo(
        video_id=path.stem,
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        width=width,
        height=height,
    )


def iter_frame_ids(
    path: str | Path,
    frame_ids: list[int] | tuple[int, ...],
) -> Iterator[tuple[int, object]]:
    """Decode a temporal window using a sequential H.264 decoder pass.

    OpenCV's CAP_PROP_POS_FRAMES seek is not reliable for every H.264 source:
    some files return False even though the decoder can read the stream. Since
    temporal localization already operates on a small local window, we avoid
    frame seeking entirely and decode from the beginning until the requested
    window is reached. This is slower for long videos, but robust and avoids
    turning a decoder warning into a pipeline failure.
    """
    unique_ids = sorted(set(int(i) for i in frame_ids if int(i) >= 0))
    if not unique_ids:
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        wanted = set(unique_ids)
        first_id = unique_ids[0]
        last_id = unique_ids[-1]
        current_id = 0

        while current_id <= last_id:
            ok, frame = cap.read()
            if not ok:
                break
            if current_id >= first_id and current_id in wanted:
                yield current_id, frame
            current_id += 1
    finally:
        cap.release()


def sample_frame_ids(frame_count: int, step: int) -> list[int]:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if step <= 0:
        raise ValueError("step must be positive")
    return list(range(0, frame_count, step))
