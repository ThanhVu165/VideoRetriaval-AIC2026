from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _candidate_paths(root: Path, video_id: str, filename: str, suffix: str) -> Iterable[Path]:
    stem = Path(filename).stem
    name = f"{stem}{suffix}"
    yield root / video_id / name
    yield root / "objects" / video_id / name
    yield root / video_id / "objects" / name


def resolve_object_path(objects_dir: str | Path | None, video_id: str, image_path: str | Path) -> str:
    if not objects_dir:
        return ""
    root = Path(objects_dir)
    if not root.exists():
        return ""
    image = Path(image_path)
    for candidate in _candidate_paths(root, video_id, image.name, ".json"):
        if candidate.is_file():
            return str(candidate)
    for candidate in root.rglob(image.stem + ".json"):
        if candidate.is_file() and candidate.parent.name == video_id:
            return str(candidate)
    return ""


def load_metadata_text(media_info_dir: str | Path | None, video_id: str) -> str:
    if not media_info_dir:
        return ""
    root = Path(media_info_dir)
    candidates = (
        root / f"{video_id}.json",
        root / "media-info" / f"{video_id}.json",
        root / video_id / "media_info.json",
        root / video_id / f"{video_id}.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False)
        except (OSError, ValueError, TypeError):
            continue
    return ""
