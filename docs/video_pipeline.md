# AIC 2026 Video Processing Pipeline

This document defines the implementation layer beyond the supplied keyframe/CLIP corpus.

## Source of truth

The official competition data are the source videos. BTC-supplied keyframes, CLIP features, object detections, and metadata are supporting evidence. The system therefore keeps the original video frame index separate from keyframe ordinal/index.

## Offline dataset index

The supplied BTC corpus is organized per video:

```text
data/
├── clip/clip-features-32/<video_id>.npy
├── mapping/map-keyframes/<video_id>.csv
└── keyframes/keyframes/<video_id>/<n>.jpg
```

Each CLIP file is `N x 512` and is aligned row-for-row with the mapping CSV and numeric keyframe filenames. The mapping's `frame_idx` is the original source-video frame ID and must not be replaced by the keyframe ordinal `n`.

Build the unified manifest and embedding matrix directly from the BTC corpus:

```bash
python -m aic2026 dataset-index \
  --clip-dir data/clip/clip-features-32 \
  --mapping-dir data/mapping/map-keyframes \
  --keyframes-dir data/keyframes/keyframes \
  --output-manifest artifacts/clip_frames_manifest.parquet \
  --output-embeddings artifacts/clip_frames.npy \
  --report-output artifacts/clip_frames.report.json
```

The builder validates, for every video:

```text
CLIP rows == mapping rows == keyframe images
```

and preserves:

```text
video_id
keyframe_idx (= mapping.n)
original_frame_id (= mapping.frame_idx)
pts_time
fps
image_path
```

The source CLIP files are float16; the generated retrieval matrix is float32 for numerical stability and FAISS compatibility.

## Source-video manifest

Source videos are scanned recursively, so nested layouts such as `data/videos/video/*.mp4` are supported:

```bash
python -m aic2026 video-manifest \
  --video-dir data/videos \
  --output artifacts/video_manifest.parquet
```

## Persistent retrieval index

After building the unified dataset index, build FAISS once:

```bash
python -m aic2026 build-index \
  --manifest artifacts/clip_frames_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
  --output-index artifacts/clip_frames.faiss \
  --metadata-output artifacts/clip_frames.index.json
```

FAISS stores vectors only; the unified manifest remains the canonical row-to-video/frame mapping. This avoids duplicated metadata and protects the distinction between `keyframe_idx` and `original_frame_id`.

FAISS is optional. Install a compatible `faiss-cpu` build when the persistent index path is used; the NumPy backend remains available for environments without FAISS.

## Online

```text
Natural-language query
  -> query embedding adapter
  -> CLIP candidate retrieval (many frames)
  -> multi-frame video evidence aggregation
  -> object + metadata reranking
  -> coarse temporal window
  -> original-video frame decoding
  -> fine frame scoring
  -> semantic keyframe
  -> final ranking / Top-k
```

The end-to-end orchestration lives in `aic2026.pipeline.AICPipeline`.

For repeated inference, reuse the persisted FAISS index:

```bash
python -m aic2026 retrieve \
  --manifest artifacts/clip_frames_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
  --faiss-index artifacts/clip_frames.faiss \
  --videos-dir data/videos \
  --query "a person enters a room" \
  --query-embedding artifacts/query_embedding.npy \
  --media-info-dir data/media_info \
  --top-k 100 \
  --radius-frames 24 \
  --max-decode-frames 96 \
  --output artifacts/candidates.json
```

The query embedding is an explicit input. The command does not silently choose a query encoder that may be incompatible with the BTC-provided CLIP features.

## Fine temporal scoring

`AICPipeline.localize()` accepts a `frame_scorer` callback. `aic2026.clip_runtime.OpenCLIPRuntime` is an optional adapter that can encode a text query and score decoded frames with a configurable ViT-B/32 checkpoint.

For reproducibility, the checkpoint used for fine scoring should be made compatible with the checkpoint that generated the supplied BTC frame features whenever possible.

## Learned reranking

`aic2026.learned_reranker.PairwiseLinearRanker` provides a dependency-free RankNet-style training primitive over retrieval, multimodal and temporal evidence. It is intentionally not hard-wired into inference because official relevance/ground-truth labels and their schema have not been supplied. Once labels are available, the learned scorer can replace or ensemble the fixed ranking weights without changing candidate generation.

`aic2026.metrics` provides Recall@k and reciprocal-rank primitives for experiments targeting Top-1/5/20/50/100 behavior.

## TRAKE

`aic2026.alignment.monotonic_event_alignment()` performs ordered event-to-frame assignment with dynamic programming. It enforces monotonic temporal order and supports a minimum frame separation.

## Q&A

`aic2026.vqa` defines a model-agnostic `VLMAnswerer` adapter. The retrieval/localization pipeline produces the visual evidence; a selected VLM can then answer the question from those frames.

## Ranking

`aic2026.ranking` keeps retrieval, temporal and multimodal evidence separate. This is intentional: later experiments can replace the fixed fusion with learned-to-rank models without changing the data flow.

KIS consumes `(video_id, frame_id)`, Q&A adds an answer, and TRAKE produces an ordered sequence of semantic keyframes. The official BTC scoring/submission implementation should only be added from the AIC 2026 query/ground-truth/submission specification.

## Current code

- `aic2026/video.py`: source-video probing and exact original-frame decoding.
- `aic2026/video_manifest.py`: recursive source-video manifest generation.
- `aic2026/dataset_index.py`: unified BTC CLIP/keyframe/mapping manifest and embedding matrix.
- `aic2026/temporal.py`: temporal-window aggregation and frame refinement.
- `aic2026/retrieval.py`: BTC CLIP frame/video candidate retrieval plus persistent FAISS loading.
- `aic2026/index_builder.py`: offline FAISS index construction.
- `aic2026/multimodal.py`: object/metadata auxiliary reranking.
- `aic2026/pipeline.py`: multi-frame candidate generation and end-to-end temporal localization.
- `aic2026/clip_runtime.py`: optional OpenCLIP text/image scoring adapter.
- `aic2026/alignment.py`: TRAKE semantic keyframe alignment.
- `aic2026/vqa.py`: VLM answerer contract.
- `aic2026/ranking.py`: final ranking and deterministic Top-k selection.
- `aic2026/learned_reranker.py`: optional pairwise learned ranking primitive.
- `aic2026/metrics.py`: ranking evaluation primitives.

Competition videos and generated indexes remain outside Git.
