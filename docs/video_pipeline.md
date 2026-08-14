# AIC 2026 Video Processing Pipeline

This document defines the first implementation layer beyond the supplied keyframe/CLIP corpus.

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

The current implementation provides deterministic video probing and original-frame decoding. Shot-boundary detection and learned temporal scoring are deliberately model-agnostic and will be added after the baseline is benchmarked.

## Online

```text
Natural-language query
  -> CLIP/BTC candidate retrieval
  -> video ranking
  -> temporal window generation
  -> fine frame scoring
  -> semantic keyframe
```

KIS consumes `(video_id, frame_id)`, Q&A adds an answer, and TRAKE produces an ordered sequence of semantic keyframes. The official BTC scoring package should only be implemented from the eventual AIC 2026 query/ground-truth/submission specification.

## Current code

- `aic2026/video.py`: source-video probing and exact original-frame decoding.
- `aic2026/video_manifest.py`: batch probe for a video directory.
- `aic2026/temporal.py`: generic temporal-window aggregation and frame refinement.
- `aic2026/retrieval.py`: existing BTC CLIP frame/video candidate retrieval baseline.
- `aic2026/build_index.py`: existing ordered CLIP matrix construction.

Competition videos and generated indexes remain outside Git.
