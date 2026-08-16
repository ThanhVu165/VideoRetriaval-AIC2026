# VideoRetrieval-AIC2026

AIC 2026 Video Retrieval pipeline: dataset integrity → multimodal candidate retrieval → temporal localization → semantic keyframe alignment → VQA adapter → ranking/top-k output.

**Project context:** read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) before making architectural changes. It is the frozen engineering direction for AI agents.

The implementation is modular. Competition data and generated artifacts stay outside Git.

## Pipeline

```text
Natural-language query
        │
        ▼
Query embedding / OpenCLIP adapter
        │
        ▼
CLIP frame retrieval
        │
        ▼
Candidate video generation (Top-N)
        │
        ├── object evidence
        ├── media metadata
        └── CLIP similarity
        │
        ▼
Multimodal reranking
        │
        ▼
Coarse temporal window
        │
        ▼
Original-video frame decoding
        │
        ▼
Fine frame scoring
        │
        ▼
Temporal event scoring + semantic peak frame
        │
        ├── Textual KIS → video + semantic keyframe
        ├── Q&A        → video + keyframe + VLM answer
        └── TRAKE      → ordered semantic keyframes
        │
        ▼
Final ranking → Top-k candidate output
```

## Dataset layout

```text
data/
├── videos/          # official source videos; keep outside Git
├── keyframes/       # supplied keyframes
├── clip/            # supplied CLIP features
├── mapping/         # keyframe → original frame / timestamp mapping
├── media_info/      # supplied video metadata
└── objects/         # supplied object detections
```

The unified manifest preserves the distinction between `keyframe_idx` and the original video `frame_id`. This distinction is required for temporal localization and final frame selection.

## Phase 0 — Dataset integrity

```bash
python -m aic2026.audit --config configs/example.yaml
python -m aic2026.validate --manifest artifacts/dataset_manifest.parquet
```

The manifest joins keyframe images, original `frame_id`, `pts_time`, `fps`, CLIP row, and object JSON path.

## Phase 1 — CLIP candidate retrieval

```bash
python -m aic2026.build_index \
  --manifest artifacts/dataset_manifest.parquet \
  --clip-dir data/clip/clip-features-32 \
  --output artifacts/clip_frames.npy \
  --report artifacts/clip_index_report.json
```

`FrameIndex` accepts a precomputed query embedding and supports frame-level retrieval plus video aggregation. NumPy inner-product search is the default; FAISS is optional.

## Phase 2 — Multimodal reranking

`aic2026.multimodal.MultimodalReranker` fuses:

- CLIP similarity — primary signal
- object-detection JSON evidence — auxiliary signal
- media-info text — auxiliary signal

The fixed weighted fusion is isolated so it can later be replaced by a learned reranker without changing the pipeline interface.

## Phase 3 — Fine temporal localization

`aic2026.video` probes the official source video and decodes **original frame IDs**. `aic2026.pipeline.AICPipeline` retrieves a coarse candidate, decodes a local frame window, optionally applies a dense frame scorer, selects the peak semantic frame, groups temporally adjacent evidence and computes a local event score.

`aic2026.clip_runtime.OpenCLIPRuntime` provides an optional PyTorch/OpenCLIP adapter for query encoding and dense frame scoring. The checkpoint remains configurable; the exact compatible ViT-B/32 checkpoint should be used when reproducing supplied BTC CLIP features.

The fine scorer is deliberately injected through an adapter so a stronger video-language encoder can replace CLIP without changing retrieval or output contracts.

## Phase 4 — TRAKE alignment

`aic2026.alignment.monotonic_event_alignment` solves ordered event-to-frame assignment with dynamic programming. It enforces temporal monotonicity and supports a minimum frame separation between events.

```text
Event 1 → keyframe 1
Event 2 → keyframe 2
...
Event N → keyframe N
```

## Phase 5 — Q&A

`aic2026.vqa` defines a model-agnostic `VLMAnswerer` adapter. A selected local VLM can be connected once the inference checkpoint and official query format are fixed.

## Phase 6 — Ranking / Top-k

`aic2026.ranking` keeps retrieval, temporal and multimodal evidence separate and produces a final `rank_score`. `top_k_submission()` provides deterministic Top-k candidate selection without claiming to implement an official BTC submission schema that has not been supplied to the repository.

## Competition scoring foundation

The supplied BTC PDF has now been encoded as pure scoring primitives in `aic2026.competition_metrics` and documented in [`docs/competition_scoring.md`](docs/competition_scoring.md). The module implements the verified Textual KIS, Q&A, TRAKE R-Score rules and the official `R@1/R@5/R@20/R@50/R@100 → Final Score` aggregation.

It intentionally does **not** invent a query-file, ground-truth-file or submission-file schema. Those parsers will be added only when the actual BTC package is available.

Run the scoring unit tests with:

```bash
python -m unittest tests.test_competition_metrics
```

## End-to-end execution

With a precomputed query embedding:

```bash
python -m aic2026 retrieve \
  --manifest artifacts/dataset_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
  --videos-dir data/videos \
  --query "a person enters a room" \
  --query-embedding artifacts/query_embedding.npy \
  --output artifacts/results.json \
  --top-k 100 \
  --radius-frames 24 \
  --max-decode-frames 96
```

For model-backed inference, use `aic2026.inference` with `OpenCLIPRuntime`; the lightweight core requirements intentionally do not force PyTorch/OpenCLIP installation.

## Current implementation status

Implemented in the active development branch:

- dataset audit and unified manifest
- CLIP frame index and video candidate generation
- source-video probing and exact original-frame decoding
- temporal window grouping/refinement
- multimodal auxiliary reranking
- coarse-to-fine pipeline orchestration
- dense fine-frame scoring adapter
- temporal event scoring and semantic peak-frame selection
- optional OpenCLIP query/frame runtime
- TRAKE monotonic semantic keyframe alignment
- VQA model adapter contract
- final candidate ranking and deterministic Top-k selection
- benchmark and ground-truth template utilities
- verified AIC2026 competition scoring primitives and tests

The next implementation step is to connect the verified scoring primitives to a real local benchmark once the official query/ground-truth package is available, then measure the existing retrieval/temporal pipeline against it. Do not jump to new model stacks before that measurement path is executable.

## Development

Competition data, embeddings, archives and generated artifacts must not be committed to Git. See `.gitignore`.
