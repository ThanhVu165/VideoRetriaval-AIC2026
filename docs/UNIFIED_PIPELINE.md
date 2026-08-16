# Unified AIC2026 Retrieval Pipeline

## 1. Source-of-truth rule

- Official source: videos.
- BTC supporting data: keyframes, CLIP ViT-B/32 features, Faster R-CNN/OpenImages object JSON, video metadata.
- Local derived evidence: ASR/OCR/caption SQLite artifacts.
- Organizer query translation: use the supplied English `Trans` field when available; fuse it with the original query rather than discarding the original.

## 2. Offline data layer

```text
Videos
  └── source-video manifest

Keyframes + mapping + BTC CLIP
  └── row-aligned unified manifest
       ├── video_id
       ├── keyframe_idx
       ├── original_frame_id
       ├── pts_time / fps
       ├── image_path
       └── object_path

Unified CLIP matrix
  └── FAISS inner-product frame index

Media metadata
  └── video-level text evidence

Objects
  └── frame-level object evidence

ASR / OCR / Caption SQLite
  └── textual/timestamp evidence stores
```

The index builder must fail on CLIP/mapping/keyframe alignment errors. Never silently shift rows.

## 3. Online query pipeline

```text
Natural-language query
        │
        ├── original language query
        └── organizer English translation (optional)
        │
        ▼
CLIP query encoder (OpenAI ViT-B/32 QuickGELU)
        │
        ▼
Frame candidate generation
        │
        ├── CLIP/FAISS
        └── optional BEiT-3/FAISS
        │
        ▼
Aggregate frame candidates → video candidates
        │
        ▼
Multimodal evidence enrichment
        ├── Objects
        ├── Metadata
        ├── Caption
        ├── OCR
        └── ASR
        │
        ▼
Rank-level / multimodal fusion
        │
        ▼
Video ranking
        │
        ▼
Original-video temporal localization
        │
        ▼
Fine frame scoring
        │
        ▼
Temporal event score + semantic keyframe
        │
        ├───────────────┬────────────────┐
        ▼               ▼                ▼
       KIS             Q&A             TRAKE
        │               │                │
 video + frame   video + frame +   video + ordered
                 semantic answer      event frames
        │               │                │
        └───────────────┴────────────────┘
                        ▼
              Competition-aware ranking
                        ▼
                  Top 100 candidates
```

## 4. Ranking policy

Default pipeline weights are isolated in `aic2026.ranking` and currently favor retrieval plus multimodal evidence while retaining temporal evidence:

```text
retrieval  = 0.45
multimodal = 0.40
temporal   = 0.15
```

These are engineering defaults, not official BTC constants. They must be tuned only through benchmark evidence.

## 5. Query translation

When the official spreadsheet provides `Trans`, encode both texts with the same OpenAI QuickGELU CLIP text encoder:

```text
query_embedding = normalize(0.20 * Vietnamese + 0.80 * English)
```

The English translation is also passed to lexical/evidence scoring. This is a retrieval aid, not a replacement for the original query.

## 6. Evidence behavior

Each candidate is enriched with:

- exact object JSON path resolved from `video_id + image_path`;
- video metadata text when the corresponding JSON exists;
- ASR/OCR/caption evidence when the SQLite artifact exists.

Unavailable modalities are omitted. The CLI reports requested and actually available evidence modalities.

## 7. Temporal behavior

The source video is decoded around the coarse candidate frame. The final output uses the original source-video frame index, not the keyframe ordinal.

`--fine-score` applies a dense CLIP frame scorer to the local temporal window. The highest-scoring semantic peak is then combined with temporally contiguous evidence to form the event score/window.

This stage cannot compensate for a video absent from candidate generation.

## 8. Optional BEiT-3

BEiT-3 is an additional candidate-generation channel, not a replacement for CLIP. It is enabled with `--beit3` and fused at video level. It must be benchmarked independently before changing the default production ranking.

## 9. Evaluation contract

The verified competition scoring uses:

```text
R@1, R@5, R@20, R@50, R@100
Final Score = mean of the five cutoffs
```

Therefore the system should retain up to 100 ranked candidates. Candidate diversity and rank placement matter; a single top-1 prediction is insufficient as the optimization target.

## 10. Regression protocol

For every pipeline change:

```text
Run baseline
  ↓
Run unified pipeline
  ↓
Compare R@1/R@5/R@20/R@50/R@100
  ↓
Inspect failures manually
  ↓
Keep only measurable improvements
```

For queries with no ground truth, manually inspect at least the first 20 candidates and record whether the expected video family enters the candidate set before spending effort on temporal/VQA improvements.
