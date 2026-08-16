# VideoRetrieval-AIC2026

Pipeline for AIC 2026 Video Retrieval: dataset audit, CLIP retrieval baseline, temporal localization, VQA, TRAKE alignment, ranking, and evaluation.

> **AI agent context:** Before making architectural or major implementation changes, read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md). It is the frozen engineering direction for this repository. The current project is a foundation, not yet competition-complete.

## Current milestone: Phase 1 — CLIP Retrieval Baseline

The repository intentionally does **not** contain competition data. Put the downloaded ZIP archives outside Git and point the audit/index scripts at the extracted dataset root.

Expected components:

```text
data/
├── keyframes/
├── clip/
├── mapping/
├── media_info/
└── objects/
```

The latest supplied full audit reports 177,321 keyframes across 873 videos, with 873 CLIP feature files, 873 mapping CSVs, 873 media-info JSONs, and 177,321 object JSONs. The project also contains an older partial audit (7,800 keyframes / 29 videos); use the latest full audit when validating the local dataset.

## Frozen engineering direction

- **Extend the current repository; do not replace it.**
- `lducc/hcm-aic` is a **reference** for selected ideas, not the base repository.
- The next milestone is **official evaluator + local benchmark**.
- Do not jump directly to OCR/ASR/VLM expansion before evaluation and temporal benchmarking are in place.
- Do not invent official query, ground-truth, scoring, or submission formats; inspect the supplied AIC/BTC specification first.

See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for the complete frozen plan, constraints, current gaps, implementation order, and definition of done.

## Phase 0 — Dataset integrity

```bash
python -m aic2026.audit --config configs/example.yaml
python -m aic2026.validate --manifest artifacts/dataset_manifest.parquet
```

Outputs:

```text
artifacts/
├── dataset_manifest.csv
├── dataset_manifest.parquet
├── dataset_audit.json
└── audit_report.txt
```

The manifest joins keyframe images, original video `frame_id`, `pts_time`, `fps`, CLIP row, and object JSON path. The original frame index is kept separate from the keyframe ordinal.

## Phase 1 — CLIP frame retrieval

Build an ordered CLIP matrix from the unified manifest:

```bash
python -m aic2026.build_index \
  --manifest artifacts/dataset_manifest.parquet \
  --clip-dir data/clip/clip-features-32 \
  --output artifacts/clip_frames.npy \
  --report artifacts/clip_index_report.json
```

Run the retrieval engine from Python:

```python
import numpy as np
from aic2026.retrieval import FrameIndex

index = FrameIndex.from_files(
    "artifacts/dataset_manifest.parquet",
    "artifacts/clip_frames.npy",
)
query_embedding = np.load("query_embedding.npy")

frames = index.search_frames(query_embedding, top_k=100)
videos = index.search_videos(
    query_embedding,
    top_k_frames=200,
    top_k_videos=100,
    aggregation="max",
)
```

The retrieval engine deliberately accepts a **precomputed query embedding**. This avoids silently assuming an unofficial query-file format or text-encoder checkpoint. The official query/ground-truth package must be inspected before implementing the AIC evaluator and submission writer.

FAISS is an optional backend; the default implementation uses normalized NumPy inner-product search. No FAISS dependency is required for the baseline.

### What Phase 1 does and does not claim

Phase 1 is a **candidate-generation baseline**, not the official AIC evaluator. It provides frame retrieval, video aggregation, and preservation of the original `frame_id`. The official scoring implementation will be added only after the BTC/AIC query, ground-truth, and submission specification is available in the project sources.

## Frozen roadmap

1. Phase 0 — dataset integrity and unified manifest
2. Phase 1 — CLIP retrieval baseline
3. **Phase A — official evaluator + local benchmark**
4. **Phase B — Textual KIS end-to-end**
5. **Phase C — fine-grained temporal localization**
6. **Phase D — TRAKE semantic keyframe alignment**
7. **Phase E — multimodal retrieval / reranking**
8. **Phase F — Q&A / VLM answer extraction**
9. **Phase G — Top-100 / Final Score optimization**
10. **Phase H — deterministic submission + regression suite**

The project is not considered competition-ready until Textual KIS, Q&A, and TRAKE all have executable end-to-end paths and are evaluated using the official R@1/R@5/R@20/R@50/R@100 and Final Score specification.

Competition data, embeddings, archives, and generated artifacts should not be committed to Git. See `.gitignore`.
