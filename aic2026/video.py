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
    """Decode a small temporal window without forcing a full-video scan.

    Some H.264 files reject CAP_PROP_POS_FRAMES seeks. We therefore try a
    frame seek first, then a timestamp seek with a small safety margin, and
    only fall back to decoding from frame 0 when both seeks are unavailable.
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
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        # Fast path: direct frame seek. Do not trust the boolean alone;
        # OpenCV documents that set() returning true only means the property
        # is supported, not that the requested position was accepted.
        frame_seek_ok = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, first_id))
        pos = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_id = int(round(pos)) if pos >= 0 else -1

        # H.264 fallback: seek by timestamp slightly before the target and
        # decode forward. This is normally much cheaper than decoding from 0.
        if (not frame_seek_ok) or current_id < 0 or abs(current_id - first_id) > max(5, int(fps)):
            margin_sec = 1.0
            target_sec = max(0.0, first_id / fps - margin_sec)
            time_seek_ok = bool(cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0))
            pos = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
            candidate_id = int(round(pos)) if pos >= 0 else -1
            if time_seek_ok and 0 <= candidate_id <= first_id:
                current_id = candidate_id
            else:
                # Last-resort robust path for unusual files/backends.
                cap.release()
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot reopen video: {path}")
                current_id = 0

        while current_id <= last_id:
            ok, frame = cap.read()
            if not ok:
                break

            # OpenCV's reported position is backend-dependent, so maintain our
            # frame counter and emit only the requested presentation indices.
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
