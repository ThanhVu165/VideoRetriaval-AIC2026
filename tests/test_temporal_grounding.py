import numpy as np

from aic2026.temporal_grounding import CLIPTemporalGrounder


class FakeRuntime:
    def score_frames(self, frames, embedding):
        assert embedding.shape == (3,)
        return [float(i) for i in range(len(frames))]


def test_clip_temporal_grounder_reuses_query_embedding():
    grounder = CLIPTemporalGrounder(FakeRuntime())
    scorer = grounder.scorer(np.array([3.0, 0.0, 0.0], dtype=np.float32))
    assert scorer([np.zeros((2, 2, 3), dtype=np.uint8)] * 3) == [0.0, 1.0, 2.0]
