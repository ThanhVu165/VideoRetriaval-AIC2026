from pathlib import Path

import numpy as np
import pandas as pd

from aic2026.dataset_index import build_unified_dataset


def test_build_unified_dataset_preserves_row_alignment(tmp_path: Path) -> None:
    clip = tmp_path / "clip"
    mapping = tmp_path / "mapping"
    keyframes = tmp_path / "keyframes"
    clip.mkdir()
    mapping.mkdir()
    (keyframes / "V001").mkdir(parents=True)

    np.save(clip / "V001.npy", np.array([[1, 0], [0, 1]], dtype=np.float16))
    pd.DataFrame(
        {"n": [1, 2], "pts_time": [0.0, 1.5], "fps": [30.0, 30.0], "frame_idx": [0, 45]}
    ).to_csv(mapping / "V001.csv", index=False)
    (keyframes / "V001" / "001.jpg").write_bytes(b"x")
    (keyframes / "V001" / "002.jpg").write_bytes(b"x")

    manifest = tmp_path / "manifest.parquet"
    embeddings = tmp_path / "embeddings.npy"
    report = build_unified_dataset(clip, mapping, keyframes, manifest, embeddings)

    assert report["rows"] == 2
    out = pd.read_parquet(manifest)
    assert out["original_frame_id"].tolist() == [0, 45]
    assert out["keyframe_idx"].tolist() == [1, 2]
    assert np.load(embeddings).shape == (2, 2)
