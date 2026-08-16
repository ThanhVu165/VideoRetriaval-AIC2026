from __future__ import annotations

from pathlib import Path

from aic2026.query_manifest import extract_events, infer_task_type, load_query_manifest, manifest_report


ROOT = Path(__file__).resolve().parents[1]


def test_task_type_uses_only_observed_query_id_convention() -> None:
    assert infer_task_type("tkis-query-01") == "TKIS"
    assert infer_task_type("qa-query-04") == "QA"
    assert infer_task_type("trake-03") == "TRAKE"
    assert infer_task_type("vkis-09") == "VKIS"
    assert infer_task_type("other-01") == "UNKNOWN"


def test_extract_events_preserves_order() -> None:
    text = """Trong video:\nE1: Event one.\nE2: Event two.\nE3: Event three."""
    events = extract_events(text)
    assert [item["event_id"] for item in events] == ["E1", "E2", "E3"]
    assert [item["text"] for item in events] == ["Event one.", "Event two.", "Event three."]


def test_supplied_workbook_manifest_inventory() -> None:
    source = Path("/mnt/data/DanhSachTruyVanAIC_Chungket.xlsx")
    if not source.exists():
        # Repository CI does not contain competition data; this test is for the
        # supplied local workbook when it is mounted.
        return

    records = load_query_manifest(source)
    report = manifest_report(records)

    assert report["queries"] == 29
    assert report["task_counts"] == {"TKIS": 14, "QA": 6, "TRAKE": 4, "VKIS": 5}
    assert report["trake_event_counts"] == {
        "trake-01": 4,
        "trake-02": 3,
        "trake-03": 3,
        "trake-04": 4,
    }
