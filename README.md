# VideoRetrieval-AIC2026

AIC 2026 Video Retrieval pipeline: dataset integrity → multimodal candidate retrieval → temporal localization → semantic keyframe alignment → VQA adapter → ranking/top-k output.

The implementation is intentionally modular. Competition data and generated artifacts stay outside Git.

## Pipeline

```text
Natural-language query
        │
        ▼
Query embedding / query adapter
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
Fine temporal scoring / localization
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

The fixed weighted fusion is deliberately isolated so it can later be replaced by a learned reranker without changing the pipeline interface.

## Phase 3 — Temporal localization

`aic2026.video` probes the official source video and decodes **original frame IDs**. `aic2026.temporal` groups sparse evidence into candidate windows and refines the semantic frame. `aic2026.pipeline.AICPipeline` connects candidate retrieval to source-video temporal decoding.

A frame scorer can be injected as an adapter. This is where a stronger CLIP/VLM/video encoder should be attached for fine-grained localization, especially when the correct event occupies only a few frames.

## Phase 4 — TRAKE alignment

`aic2026.alignment.monotonic_event_alignment` solves ordered event-to-frame assignment with dynamic programming. It enforces temporal monotonicity and supports a minimum frame separation between events.

This gives the core alignment primitive for:

```text
Event 1 → keyframe 1
Event 2 → keyframe 2
...
Event N → keyframe N
```

## Phase 5 — Q&A

`aic2026.vqa` defines a model-agnostic `VLMAnswerer` adapter. The pipeline does not hard-code a VLM checkpoint or an unofficial query format; a local VLM can be plugged into the adapter once the intended inference model is selected.

## Phase 6 — Ranking / Top-k

`aic2026.ranking` keeps retrieval, temporal and multimodal evidence separate and produces a final `rank_score`. `top_k_submission()` provides deterministic Top-k candidate selection without pretending to implement an official BTC submission schema that has not been supplied to the repository.

## Current implementation status

Implemented in the active development branch:

- dataset audit and unified manifest
- CLIP frame index and video candidate generation
- source-video probing and frame decoding
- temporal window grouping/refinement
- multimodal auxiliary reranking
- coarse-to-fine pipeline orchestration
- TRAKE monotonic semantic keyframe alignment
- VQA model adapter contract
- final candidate ranking and Top-k selection
- unit tests for retrieval, temporal processing, alignment and ranking

The next model-level improvements should plug into these interfaces rather than redesign the data flow: stronger query encoding, learned multimodal reranking, dense frame scoring for fine temporal localization, and a selected VLM for Q&A.

## Development

Competition data, embeddings, archives and generated artifacts must not be committed to Git. See `.gitignore`.
