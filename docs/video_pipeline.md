# AIC 2026 Unified Video Processing Pipeline

The official competition source of truth is the video. BTC-supplied keyframes, CLIP features, object detections and metadata are supporting evidence. Local ASR/OCR/caption artifacts are additional evidence channels when present.

## Offline

```text
Videos
  └── source-video manifest

BTC keyframes + mapping + CLIP
  └── row-aligned unified manifest
       ├── video_id
       ├── keyframe_idx
       ├── original_frame_id
       ├── pts_time / fps
       ├── image_path
       └── object_path

CLIP matrix
  └── persistent FAISS frame index

Metadata / Objects / ASR / OCR / Caption
  └── evidence stores
```

The dataset builder enforces:

```text
CLIP rows == mapping rows == keyframe images
```

for every video and preserves `mapping.frame_idx` as the original source-video frame ID.

## Online

```text
Natural-language query
  → original + organizer translation
  → CLIP query embedding
  → CLIP frame candidate generation
  → optional BEiT-3 candidate generation
  → video aggregation
  → Objects + Metadata + ASR/OCR/Caption enrichment
  → multimodal/rank-level fusion
  → video ranking
  → coarse temporal window
  → original-video decoding
  → fine frame scoring
  → semantic keyframe
  → KIS / Q&A / TRAKE output
  → Top-100 ranking
```

## Commands

```bash
python -m aic2026 dataset-index \
  --clip-dir data/clip/clip-features-32 \
  --mapping-dir data/mapping/map-keyframes \
  --keyframes-dir data/keyframes/keyframes \
  --objects-dir data/objects \
  --output-manifest artifacts/clip_frames_manifest.parquet \
  --output-embeddings artifacts/clip_frames.npy

python -m aic2026 build-index \
  --manifest artifacts/clip_frames_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
  --output-index artifacts/clip_frames.faiss
```

Example retrieval:

```bash
python -m aic2026 retrieve \
  --manifest artifacts/clip_frames_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
  --faiss-index artifacts/clip_frames.faiss \
  --videos-dir data/videos \
  --query "..." \
  --query-translated "..." \
  --top-k 100 \
  --fine-score \
  --output artifacts/validation/query.json
```

## Temporal localization

`aic2026.pipeline.AICPipeline` resolves source videos recursively and decodes original frame IDs around a coarse candidate. `--fine-score` injects a dense CLIP frame scorer. The semantic keyframe is selected from temporal evidence, not from the keyframe ordinal.

## TRAKE / Q&A

TRAKE uses ordered event alignment and semantic keyframes. Q&A uses the retrieval/localization output as evidence for a VLM answerer. Both share the same candidate-generation and temporal foundation.

## Ranking

The fixed engineering weights are currently:

```text
retrieval  0.45
multimodal 0.40
temporal   0.15
```

These are not BTC constants. They are tunable engineering parameters and must be validated against the competition objective.

The official scoring primitives are in `aic2026.competition_metrics`; they implement the verified R@1/R@5/R@20/R@50/R@100 and Final Score rules without inventing an unverified submission schema.
