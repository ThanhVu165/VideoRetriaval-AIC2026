# AIC 2026 Query / Evidence Source Audit

Frozen audit date: 2026-08-16.

This document records what is actually present in the supplied local materials. It does **not** define an official BTC submission or ground-truth schema.

## 1. Query workbook

Source: `DanhSachTruyVanAIC_Chungket.xlsx`, sheet `Sheet1`.

The workbook contains 29 query records with columns:

```text
Query Name
Description
Trans
```

Observed task inventory:

| Prefix | Count | IDs present |
|---|---:|---|
| `tkis-query-*` | 14 | 01–10, 12–15 |
| `qa-query-*` | 6 | 01, 02, 04–07 |
| `trake-*` | 4 | 01–04 |
| `vkis-*` | 5 | 01, 02, 06, 07, 09 |
| **Total** | **29** | |

Important: the workbook is a **query-description source**. It does not, by itself, contain the video ID, ground-truth frame interval, or official submission format required for competition scoring.

### TRAKE event structure

The four TRAKE queries explicitly contain ordered event labels:

- `trake-01`: 4 events
- `trake-02`: 3 events
- `trake-03`: 3 events
- `trake-04`: 4 events

These event descriptions are suitable as event-query input to the existing TRAKE baseline. They are not ground-truth frame intervals.

### Q&A structure

The six Q&A queries contain natural-language questions and constraints on the expected answer. Examples include counting players in a penalty area, counting pieces of a food item, identifying a festival city, identifying a painter surname, identifying a company, and identifying a fashion brand.

The workbook therefore provides the **question/query side** of Q&A, but not the official semantic-answer evaluator.

## 2. Existing frame index

The supplied `frames.csv` contains 177,321 rows and the following fields:

```text
frame_id
frame_uid
video_id
keyframe_n
timestamp_sec
frame_path
source
```

This is sufficient to establish a dataset-level keyframe identity and timestamp table. It is not a task ground-truth table.

The repository must continue to distinguish `keyframe_n` / keyframe identity from the original video `frame_id`.

## 3. Caption evidence database

`caption.sqlite` contains a primary `caption` table with:

```text
video_id
timestamp_sec
frame_path
text
```

Observed row count: **177,321**.

The database also contains the expected SQLite FTS auxiliary tables (`caption_data`, `caption_idx`, `caption_content`, `caption_docsize`, `caption_config`).

This is an **evidence/index source**, not ground truth.

## 4. OCR evidence database

`ocr.sqlite` contains a primary `ocr` table with:

```text
video_id
timestamp_sec
frame_path
text
```

Observed row count: **164,820**.

It also contains SQLite FTS auxiliary tables.

This is an **evidence/index source**, not ground truth.

## 5. Supplied dataset audit

The latest full dataset audit reports:

- 177,321 keyframes
- 873 videos represented by the supplied CLIP/mapping/media-info/object assets
- 873 CLIP feature files
- 873 mapping CSV files
- 873 media-info JSON files
- 177,321 object JSON files

The official BTC PDF states that the source videos are the official competition data and that keyframes, objects, CLIP features and metadata are supporting data.

## 6. Ground-truth boundary

The repository currently has a local development GT contract used by its benchmark utilities. It is **not** the organizer's official GT package.

The existing local evaluator code also exposes fields such as:

```text
query_id
video_ids
frames
intervals
events
answer
```

Those fields are useful for local development, but they must not be promoted to an official schema without the actual BTC package.

Therefore:

```text
Workbook query descriptions       -> available
Frame/timestamp evidence          -> available
Caption evidence                  -> available
OCR evidence                      -> available
Official query -> GT binding      -> not verified
Official submission schema        -> not verified
Official Q&A semantic matcher     -> not verified
Official TRAKE GT package         -> not verified
```

## 7. What we can implement now

Without inventing organizer formats, the repository can safely implement:

1. A canonical query loader for the workbook's `Query Name` / `Description` / `Trans` columns.
2. Task classification based on the observed query IDs (`tkis`, `qa`, `trake`, `vkis`) as a **local dataset convention**.
3. Event extraction for the explicit `E1:`, `E2:`, ... TRAKE descriptions.
4. Retrieval over the existing CLIP index.
5. Evidence retrieval over caption/OCR databases.
6. Local KIS benchmark evaluation when explicit local GT intervals are supplied.
7. Baseline TRAKE and VQA inference using the already implemented repository interfaces.

## 8. What must remain blocked

Do not implement any of the following as "official" until the corresponding organizer source is available:

- official GT parser;
- official submission serializer;
- official Q&A semantic answer matcher;
- official TRAKE evaluator;
- official query-to-video/frame annotations.

## 9. Next implementation target

The next concrete engineering step is **not another model**.

It is to make the workbook a first-class query input and produce a reproducible local benchmark manifest:

```text
Workbook
  -> canonical query records
  -> task type
  -> language variants
  -> TRAKE event list (when present)
  -> benchmark input
```

Then connect that manifest to the existing retrieval/temporal pipeline and measure the current baseline before introducing additional multimodal models.
