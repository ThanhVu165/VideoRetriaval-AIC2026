import numpy as np

from aic2026.fine_scoring import select_peak_frame, temporal_event_score


def test_select_peak_frame():
    assert select_peak_frame([10, 11, 12], [0.1, 0.9, 0.2]) == 11


def test_temporal_event_score_rewards_local_continuity():
    isolated = temporal_event_score([1, 2, 3], [0.0, 1.0, 0.0])
    continuous = temporal_event_score([1, 2, 3], [0.8, 1.0, 0.7])
    assert continuous > isolated


def test_normalized_score_is_finite_for_constant_values():
    from aic2026.fine_scoring import normalize_scores

    result = normalize_scores(np.array([2.0, 2.0]))
    assert np.all(np.isfinite(result))
