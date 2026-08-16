# AIC2026 Video Retrieval — Project Context

> Persistent project context for AI-assisted development. Update this file when the local dataset, artifacts, pipeline, or evaluation baseline materially changes.
> Snapshot date: 2026-08-16.

## 1. Local repository

- Windows project root: `C:\VideoRetrieval-AIC2026\VideoRetriaval-AIC2026`
- GitHub repository: `ThanhVu165/VideoRetriaval-AIC2026`
- Default branch: `main`

## 2. Local data layout

The current local `data/` directory contains:

```text
data/
├── clip/
├── keyframes/
├── mapping/
├── media_info/
├── objects/
├── queries/
├── videos/
└── ground_truth.json
```

Roles:
- `clip/`: CLIP features.
- `keyframes/`: extracted/provided keyframes.
- `mapping/`: keyframe-to-original-video frame mapping.
- `media_info/`: video/media metadata.
- `objects/`: object-detection artifacts.
- `queries/`: query datasets/spreadsheets/text.
- `videos/`: official/source videos.
- `ground_truth.json`: local ground-truth/annotation artifact.

The source videos are present locally under `data/videos/video/`. Example files include `L21_V001.mp4`, `L21_V002.mp4`, etc.

## 3. Current frame retrieval index — confirmed

### `artifacts/clip_frames.index.json`

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

### `artifacts/clip_frames.report.json`

```json
{
  "videos_indexed": 873,
  "rows": 177321,
  "dimension": 512,
  "source_dtype": "float16",
  "index_dtype": "float32",
  "clip_dir": "data\\clip\\clip-features-32",
  "mapping_dir": "data\\mapping\\map-keyframes",
  "keyframes_dir": "data\\keyframes\\keyframes",
  "manifest": "artifacts\\clip_frames_manifest.parquet",
  "embeddings": "artifacts\\clip_frames.npy"
}
```

Therefore the current confirmed baseline is:
- 873 indexed videos.
- 177,321 frame/keyframe rows.
- 512-dimensional CLIP embeddings.
- source embedding dtype: float16.
- FAISS index dtype: float32.
- cosine-equivalent inner product because embeddings are L2-normalized.
- FAISS backend: `IndexFlatIP` (exact flat search, not ANN approximation).

## 4. Frame manifest — confirmed

`artifacts/clip_frames_manifest.parquet` has 177,321 rows and these columns:

```text
video_id
keyframe_idx
original_frame_id
pts_time
fps
image_path
object_path
```

Observed examples:

```text
video_id  keyframe_idx  original_frame_id  pts_time  fps   image_path
L21_V001  1             0                  0.0       30.0  data\keyframes\keyframes\L21_V001\001.jpg
L21_V001  2             90                3.0       30.0  data\keyframes\keyframes\L21_V001\002.jpg
L21_V001  3             261               8.7       30.0  data\keyframes\keyframes\L21_V001\003.jpg
```

Important: for temporal evaluation, do not confuse `keyframe_idx` with `original_frame_id`. The correct chain is:

```text
FAISS row
  -> manifest row
  -> video_id + keyframe_idx
  -> original_frame_id
  -> pts_time
  -> source video frame
```

## 5. Current artifacts directory

The local `artifacts/` directory currently contains, at minimum:

```text
artifacts/
├── benchmark/
├── benchmark_en/
├── manual_audit/
├── validation/
├── audit_report.txt
├── caption.sqlite
├── clip.faiss
├── clip_frames.faiss
├── clip_frames.index.json
├── clip_frames.npy
├── clip_frames.report.json
├── clip_frames_manifest.parquet
├── dataset_audit.json
├── dataset_manifest.csv
├── frames.csv
├── ocr.sqlite
├── result.json
├── retrieve_error.log
├── video_manifest.json
└── video_manifest.parquet
```

SQLite evidence artifacts currently present:
- `artifacts/caption.sqlite`
- `artifacts/ocr.sqlite`

These are optional evidence sources for retrieval/reranking and must not be assumed to be the official primary dataset.

## 6. Query source

Local official query spreadsheet:

`data/queries/DanhSachTruyVanAIC_Chungket.xlsx`

Observed workbook:
- sheet: `Sheet1`
- columns: `Query Name`, `Description`, `Trans`

Observed first query IDs include:
- `tkis-query-01`
- `tkis-query-02`
- `tkis-query-03`
- `tkis-query-04`

The project also contains a test query package `query-p1-groupA.zip` in the local working environment; it is not treated as a repository artifact unless explicitly committed.

## 7. CLI currently available

`python -m aic2026 --help` currently exposes:

```text
aic2026
├── video-manifest
├── dataset-index
├── build-index
├── inspect-evidence
├── retrieve
├── trake
├── vqa
├── benchmark
└── ground-truth-template
```

Current `retrieve --help` exposes:

```text
--manifest
--embeddings
--faiss-index
--videos-dir
--query
--query-embedding
--model-name
--pretrained
--device
--output
--media-info-dir
--asr-db
--ocr-db
--caption-db
--evidence-rrf-k
--top-k
--radius-frames
--max-decode-frames
--fine-score
--fine-batch-size
```

Current intended retrieval pipeline:

```text
Natural-language query
        |
        v
Query embedding
        |
        v
CLIP frame retrieval / candidate generation
        |
        v
Multimodal evidence (optional)
        |
        v
Temporal localization
        |
        v
Fine CLIP temporal rescoring (optional)
        |
        v
Video + semantic keyframe candidate
```

## 8. Manual audit protocol

For a candidate result, inspect a neighborhood around `best_frame_id`, not only the single frame. Current baseline uses a window of approximately +/-24 original frames.

Manual audit must separate:

1. **Video retrieval correctness** — is this the correct source video?
2. **Event localization correctness** — does the temporal window contain the event described by the query?
3. **Semantic keyframe correctness** — is `best_frame_id` actually the most representative frame for that event?
4. **Semantic discrimination** — is the result genuinely the requested event rather than a visually similar event?

A high CLIP similarity score alone is not evidence that temporal localization is correct.

## 9. Baseline validation artifacts

Example:

`artifacts/validation/tkis-query-01.baseline.json`

Observed result fields include:

```text
video_id
retrieval_score
multimodal_score
best_frame_idx
best_frame_id
best_pts_time
object_path
retrieval_best_score
retrieval_topk_mean
retrieval_score_std
temporal_score
rank_score
temporal_start_frame
temporal_end_frame
semantic_keyframe
```

Baseline results must be preserved. Do not overwrite them when testing a new method.

Preferred convention:

```text
artifacts/validation/<query>.baseline.json
artifacts/validation/<query>.<experiment>.json
artifacts/manual_audit/<query>.<experiment>.*
```

## 10. Current investigation principle

Before modifying the retrieval architecture, manually validate representative failures/successes. For each test query, determine whether the error originates from:

```text
coarse retrieval
    -> temporal localization
    -> semantic keyframe selection
    -> multimodal reranking
```

Do not add a more complex model merely because a baseline score is low. First identify the failing stage.

For ranking-oriented evaluation, preserve a sufficiently large candidate set (e.g. Top-20/Top-50/Top-100 depending on experiment) so that we can distinguish candidate-generation failure from reranking failure.

## 11. Known manual validation examples already inspected

Several baseline temporal windows were manually visualized. They demonstrated why the center frame and neighboring frames must be inspected together: some candidates were semantically plausible while others corresponded to adjacent but different actions/scenes.

The audit should therefore record both the predicted frame and the event interpretation, rather than marking a result correct solely from visual similarity.

## 12. Artifact hygiene

The repository context should document artifact state, but large generated data should generally remain local unless there is a specific reason to version it.

Do not commit large binary datasets, videos, FAISS indexes, SQLite databases, or generated frame directories merely for context. Prefer small manifests, reports, schemas, configuration, and audit notes.

When an artifact changes materially, update this context with:
- artifact path;
- row/vector count;
- dimensionality/dtype where relevant;
- source directories;
- mapping semantics;
- generation command or code path if known;
- validation status.

## 13. Objective of the project

The target system is a competition-grade Video Retrieval / Video Understanding pipeline supporting:

- Textual Known Item Search (Textual KIS): retrieve the correct video and frame/event.
- Visual Question Answering (Q&A): retrieve the correct video/time and answer the question.
- Temporal Retrieval and Alignment of Key Events (TRAKE): retrieve the correct video and align each ordered event to a semantic keyframe.

The final system should optimize ranking quality as well as single-best accuracy, with attention to Top-k retrieval and fine temporal localization.
