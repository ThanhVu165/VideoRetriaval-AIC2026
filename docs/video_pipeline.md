# AIC 2026 Video Processing Pipeline

This document defines the implementation layer beyond the supplied keyframe/CLIP corpus.

## Source of truth

The official competition data are the source videos. BTC-supplied keyframes, CLIP features, object detections, and metadata are supporting evidence. The system therefore keeps the original video frame index separate from keyframe ordinal/index.

## Offline

```text
Source video
  -> probe metadata
  -> source-video manifest
  -> temporal structure
  -> candidate windows
```

`aic2026.video` provides deterministic probing and original-frame decoding. `aic2026.video_manifest` builds a batch manifest for source videos.

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

### Multi-frame candidate generation

The pipeline retrieves a larger frame pool than the final Top-k and aggregates the strongest `per_video_k` frames. The video score combines the best frame with the mean score of the strongest supporting frames. This reduces sensitivity to one noisy keyframe while retaining single-frame recall.

The strongest retrieved frame remains the coarse temporal anchor, and its `object_path` is retained for multimodal evidence.

### Running the baseline pipeline

```bash
python -m aic2026 retrieve \
  --manifest artifacts/dataset_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
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
- `aic2026/video_manifest.py`: batch source-video manifest generation.
- `aic2026/temporal.py`: temporal-window aggregation and frame refinement.
- `aic2026/retrieval.py`: BTC CLIP frame/video candidate retrieval.
- `aic2026/multimodal.py`: object/metadata auxiliary reranking.
- `aic2026/pipeline.py`: multi-frame candidate generation and end-to-end temporal localization.
- `aic2026/clip_runtime.py`: optional OpenCLIP text/image scoring adapter.
- `aic2026/alignment.py`: TRAKE semantic keyframe alignment.
- `aic2026/vqa.py`: VLM answerer contract.
- `aic2026/ranking.py`: final ranking and deterministic Top-k selection.
- `aic2026/learned_reranker.py`: optional pairwise learned ranking primitive.
- `aic2026/metrics.py`: ranking evaluation primitives.

Competition videos and generated indexes remain outside Git.
