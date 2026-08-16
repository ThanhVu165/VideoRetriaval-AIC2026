# AIC2026 Video Retrieval — Persistent Project Context

Snapshot: 2026-08-16. This file complements the canonical `PROJECT_CONTEXT.md` with the confirmed local artifact state.

## Repository

- GitHub: `ThanhVu165/VideoRetriaval-AIC2026`
- Unified development branch: `unified-aic2026-pipeline`

## Confirmed local dataset

```text
data/
├── clip/
├── keyframes/
├── mapping/
├── media_info/
├── objects/
├── queries/
└── videos/
```

Source videos are present under `data/videos/video/`.

## Confirmed retrieval artifact

`artifacts/clip_frames.index.json`:

```json
{
  "index": "artifacts\\clip_frames.faiss",
  "manifest": "artifacts\\clip_frames_manifest.parquet",
  "embeddings": "artifacts\\clip_frames.npy",
  "rows": 177321,
  "dimension": 512,
  "metric": "inner_product_on_l2_normalized_embeddings",
  "backend": "faiss.IndexFlatIP"
}
```

`artifacts/clip_frames.report.json` confirms 873 indexed videos and 177,321 rows.

## Manifest contract

`artifacts/clip_frames_manifest.parquet` columns:

```text
video_id
keyframe_idx
original_frame_id
pts_time
fps
image_path
object_path
```

`keyframe_idx` is not the original video frame ID. The authoritative temporal chain is:

```text
FAISS row → manifest row → video_id + original_frame_id → source video frame
```

## Query source

`data/queries/DanhSachTruyVanAIC_Chungket.xlsx` contains `Query Name`, `Description`, and `Trans`. The unified retrieval CLI can fuse the original query with the organizer English translation using a 20/80 weighting before normalization.

## Current validation principle

For a failed query, inspect the stages separately:

```text
candidate generation
      ↓
video ranking
      ↓
temporal localization
      ↓
semantic keyframe
```

Do not treat a high CLIP score as proof of temporal correctness. Preserve baseline artifacts and write new experiment outputs separately.

## Current artifact inventory

The local environment has included:

```text
artifacts/clip_frames.faiss
artifacts/clip_frames.npy
artifacts/clip_frames_manifest.parquet
artifacts/clip_frames.index.json
artifacts/clip_frames.report.json
artifacts/caption.sqlite
artifacts/ocr.sqlite
artifacts/validation/
artifacts/benchmark/
artifacts/manual_audit/
```

ASR/OCR/caption SQLite files are evidence channels, not the official primary dataset. They are used when available.

## Unified pipeline status

The unified branch activates:

- BTC CLIP/keyframe/mapping alignment;
- BTC Faster R-CNN object evidence;
- BTC media metadata evidence;
- ASR/OCR/caption evidence stores when available;
- organizer English query translation;
- OpenAI ViT-B/32 QuickGELU query encoding;
- optional BEiT-3 candidate generation;
- original-video temporal localization;
- fine CLIP frame scoring;
- semantic keyframe selection;
- TRAKE and VQA adapters;
- competition-aware Top-100 ranking.

Large datasets and generated binary artifacts remain outside Git.
