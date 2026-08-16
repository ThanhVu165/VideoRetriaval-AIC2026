from __future__ import annotations

from collections.abc import Callable, Sequence

KS: tuple[int, ...] = (1, 5, 20, 50, 100)

AnswerMatcher = Callable[[str, str], bool]


def kis_r_score(
    video_id: str,
    frame_id: int,
    gt_video_id: str,
    start_frame: int,
    end_frame: int,
) -> float:
    """AIC2026 Textual KIS R-Score for one submitted answer."""
    return float(video_id == gt_video_id and start_frame <= int(frame_id) <= end_frame)


def qa_r_score(
    video_id: str,
    frame_id: int,
    answer: str,
    gt_video_id: str,
    start_frame: int,
    end_frame: int,
    gt_answer: str,
    answer_matcher: AnswerMatcher,
) -> float:
    """AIC2026 Q&A R-Score with an explicit semantic-answer matcher.

    The organizer defines answer correctness semantically, but does not
    prescribe a local string-normalization algorithm. Therefore the matcher
    is injected instead of silently inventing one.
    """
    return float(
        video_id == gt_video_id
        and start_frame <= int(frame_id) <= end_frame
        and answer_matcher(str(answer), str(gt_answer))
    )


def trake_r_score(
    video_id: str,
    frame_ids: Sequence[int],
    gt_video_id: str,
    intervals: Sequence[tuple[int, int]],
) -> float:
    """AIC2026 TRAKE R-Score for one submitted sequence.

    The organizer specifies a strict video gate: a wrong video yields zero.
    When the video is correct, score is the fraction of event frames that
    fall inside their corresponding ground-truth intervals.
    """
    if video_id != gt_video_id:
        return 0.0
    if len(frame_ids) != len(intervals) or not intervals:
        return 0.0
    hits = sum(
        int(start) <= int(frame) <= int(end)
        for frame, (start, end) in zip(frame_ids, intervals)
    )
    return hits / len(intervals)


def r_at_k(r_scores: Sequence[float], k: int) -> float:
    """Top-k R-Score: maximum R-Score among the first k answers."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not r_scores:
        return 0.0
    return max(float(x) for x in r_scores[:k])


def final_score(r_scores: Sequence[float], ks: Sequence[int] = KS) -> float:
    """AIC2026 Final Score: mean of Top-k R-Scores at the official cutoffs."""
    if not ks:
        raise ValueError("ks must not be empty")
    return sum(r_at_k(r_scores, int(k)) for k in ks) / len(ks)


def ranking_profile(r_scores: Sequence[float], ks: Sequence[int] = KS) -> dict[str, float]:
    """Return R@1/R@5/R@20/R@50/R@100 and Final Score for one query."""
    result = {f"r@{int(k)}": r_at_k(r_scores, int(k)) for k in ks}
    result["final_score"] = sum(result.values()) / len(ks)
    return result
