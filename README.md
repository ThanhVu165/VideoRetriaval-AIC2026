# VideoRetriaval-AIC2026

Unified AIC 2026 Video Retrieval / Video Understanding pipeline.

**Canonical context:** [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)

## Unified architecture

```text
Official source videos + BTC support data
        │
        ├── Keyframes + original-frame mapping
        ├── BTC CLIP ViT-B/32 features
        ├── Faster R-CNN object JSON
        └── Video media metadata
        │
        ▼
Unified manifest + FAISS frame index
        │
        ▼
Query understanding
  ├── Vietnamese query
  └── organizer English translation (when supplied)
        │
        ▼
Multi-channel candidate generation
  ├── CLIP frame retrieval
  ├── optional BEiT-3 retrieval
  └── ASR / OCR / Caption evidence
        │
        ▼
Video-level multimodal reranking
  ├── CLIP score
  ├── object evidence
  ├── metadata evidence
  └── ASR/OCR/caption RRF evidence
        │
        ▼
Original-video temporal localization
        │
        ▼
Fine frame scoring / semantic keyframe
        │
        ├── Textual KIS → video + frame
        ├── Q&A        → video + frame + answer
        └── TRAKE      → video + ordered semantic keyframes
        │
        ▼
Competition-aware ranking → Top-100
```

Video remains the source of truth. Keyframes, CLIP, Objects and Metadata are BTC-provided supporting data; ASR/OCR/Caption artifacts are optional local evidence channels and are used when present.

## Dataset facts currently verified

The latest full local audit reports **873 videos / 177,321 keyframes**, with 873 CLIP feature files, 873 mapping CSVs, 873 media-info JSONs and 177,321 object JSONs. The unified manifest preserves both the keyframe ordinal and the original source-video `frame_id`.

## Main commands

### Build the unified dataset index

```bash
python -m aic2026 dataset-index \
  --clip-dir data/clip/clip-features-32 \
  --mapping-dir data/mapping/map-keyframes \
  --keyframes-dir data/keyframes/keyframes \
  --objects-dir data/objects \
  --output-manifest artifacts/clip_frames_manifest.parquet \
  --output-embeddings artifacts/clip_frames.npy \
  --report-output artifacts/clip_frames.report.json
```

### Build the persistent FAISS index

```bash
python -m aic2026 build-index \
  --manifest artifacts/clip_frames_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
  --output-index artifacts/clip_frames.faiss
```

### Unified single-query retrieval

The default retrieval path uses available BTC/support evidence. The organizer English translation can be supplied with `--query-translated`. `--fine-score` enables dense CLIP temporal rescoring. `--beit3` adds the optional BEiT-3 channel.

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

The command reports which evidence modalities were actually available, whether BTC Objects/Metadata were resolved, whether translation was used, and whether BEiT-3/fine scoring was enabled.

### Unified benchmark over the organizer query sheet

For the supplied `DanhSachTruyVanAIC_Chungket.xlsx`, use the organizer English `Trans` column and keep the official maximum of 100 candidates:

```bash
python scripts/benchmark_unified.py \
  --queries data/queries/DanhSachTruyVanAIC_Chungket.xlsx \
  --manifest artifacts/clip_frames_manifest.parquet \
  --embeddings artifacts/clip_frames.npy \
  --faiss-index artifacts/clip_frames.faiss \
  --videos-dir data/videos \
  --media-info-dir data/media_info \
  --objects-dir data/objects \
  --top-k 100 \
  --fine-score \
  --output-dir artifacts/benchmark_unified
```

Use `--task-prefix tkis-`, `--task-prefix qa-`, `--task-prefix trake-`, or `--task-prefix vkis-` to isolate a query family. This benchmark uses the same canonical retrieval/temporal pipeline as the single-query CLI instead of maintaining a separate retrieval implementation.

## Retrieval design rules

1. **Candidate generation is the first bottleneck.** The system must keep enough candidates before reranking; temporal localization cannot recover a video that never entered the candidate pool.
2. **Use rank-level fusion for heterogeneous evidence.** Raw CLIP/object/OCR/ASR/caption scores are not assumed to be numerically comparable.
3. **Use original videos for temporal truth.** Keyframes are retrieval evidence; final frame IDs come from the original video/mapping.
4. **Keep missing modalities graceful.** Metadata or an evidence database may be absent for a video/query.
5. **Do not claim BEiT-3 is better without measurement.** It is an optional channel and must be benchmarked against the CLIP baseline.
6. **Optimize the competition objective.** Evaluate R@1/R@5/R@20/R@50/R@100 and Final Score rather than only single-best retrieval.

## Competition scoring

The verified competition primitives are in `aic2026.competition_metrics` and `docs/competition_scoring.md`. They implement the verified Textual KIS, Q&A and TRAKE R-Score rules and the five-cutoff Final Score aggregation. They do not invent an official query/ground-truth/submission file schema.

## Repository policy

Do not commit competition videos, embeddings, SQLite artifacts, FAISS indexes or generated validation outputs. Keep large local artifacts under the ignored `data/` and `artifacts/` paths.
