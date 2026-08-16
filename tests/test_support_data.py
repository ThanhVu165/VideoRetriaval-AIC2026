from pathlib import Path

from aic2026.support_data import load_metadata_text, resolve_object_path


def test_resolve_object_path(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    target = root / "L23_V001" / "001.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"detection_class_names": ["person", "bicycle"]}', encoding="utf-8")
    found = resolve_object_path(root, "L23_V001", "data/keyframes/keyframes/L23_V001/001.jpg")
    assert Path(found) == target


def test_load_metadata_text(tmp_path: Path) -> None:
    root = tmp_path / "media_info"
    root.mkdir()
    (root / "L23_V001.json").write_text('{"title": "cycling race"}', encoding="utf-8")
    text = load_metadata_text(root, "L23_V001")
    assert "cycling race" in text
