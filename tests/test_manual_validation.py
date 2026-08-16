from aic2026.manual_validation import ValidationRecord, binary_r_at_k, first_valid_rank


def test_tkis_manual_validation():
    records = [
        ValidationRecord(
            query_id="tkis-query-01",
            task_type="TKIS",
            rank=5,
            video_id="v1",
            frame_id=100,
            video_match=True,
            frame_match=True,
        )
    ]
    assert first_valid_rank(records) == 5
    assert binary_r_at_k(records, 1) == 0.0
    assert binary_r_at_k(records, 5) == 1.0


def test_qa_requires_answer_match():
    records = [
        ValidationRecord(
            query_id="qa-query-01",
            task_type="QA",
            rank=1,
            video_id="v1",
            frame_id=100,
            video_match=True,
            frame_match=True,
            answer_match=False,
        ),
        ValidationRecord(
            query_id="qa-query-01",
            task_type="QA",
            rank=3,
            video_id="v1",
            frame_id=120,
            video_match=True,
            frame_match=True,
            answer_match=True,
        ),
    ]
    assert first_valid_rank(records) == 3
