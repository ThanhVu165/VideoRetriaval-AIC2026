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
    """Decode requested frame IDs with one sequential decoder pass.

    The previous implementation performed a random seek for every requested
    frame. That is expensive for H.264 and can trigger decoder warnings when
    a window contains many nearby frames. We now seek once to the first
    requested frame and decode forward, yielding only requested IDs.
    """
    unique_ids = sorted(set(int(i) for i in frame_ids if int(i) >= 0))
    if not unique_ids:
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        first_id = unique_ids[0]
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, first_id):
            raise RuntimeError(f"Cannot seek to frame {first_id}: {path}")

        wanted = set(unique_ids)
        current_id = first_id
        last_id = unique_ids[-1]

        while current_id <= last_id:
            ok, frame = cap.read()
            if not ok:
                break
            if current_id in wanted:
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
