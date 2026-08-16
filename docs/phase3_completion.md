# AIC 2026 Phase 3: task-complete baseline

This phase focuses on **functional completeness before optimization**.

## Implemented

### Textual KIS

`aic2026 retrieve --fine-score` now performs:

1. CLIP frame retrieval from the persistent FAISS index.
2. Video-level candidate aggregation and multimodal evidence fusion.
3. Original-video temporal decoding.
4. CLIP fine scoring over decoded original frames.
5. Semantic peak-frame selection and temporal window scoring.
6. Final ranking and Top-k submission output.

The fine scorer is intentionally a baseline. It is not a learned temporal model.

### TRAKE

`aic2026 trake` accepts an ordered JSON list of event descriptions and performs:

1. Independent event retrieval.
2. Cross-event video consistency voting.
3. Joint temporal-span decoding from the selected source video.
4. Event-to-frame CLIP compatibility scoring.
5. Monotonic dynamic-programming alignment.
6. One semantic keyframe per ordered event.

Example input:

```json
[
  "a person enters the room",
  "the person sits at a table",
  "the person leaves the room"
]
```

### Q&A

`aic2026 vqa` performs retrieval + temporal localization + VLM answer generation.
The VLM is provided through an optional Hugging Face `image-text-to-text`
pipeline so the repository does not force a single heavyweight checkpoint.

The output contains:

- query id
- video id
- evidence frame ids
- generated answer

## Still required before competition submission

1. Bind the official query spreadsheet schema for all task types.
2. Build/verify task-specific ground truth and evaluator outputs.
3. Validate the exact submission format required by the organizer.
4. Run all official queries end-to-end.
5. Replace the baseline CLIP temporal scorer with a stronger temporal/VLM model.
6. Train and calibrate a learned reranker on verified ground truth.
7. Add task-specific answer normalization and evaluation for Q&A.

The objective of this phase is to remove missing functionality, not to maximize
leaderboard score yet.
