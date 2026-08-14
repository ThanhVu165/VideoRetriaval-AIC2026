# VideoRetrieval-AIC2026

Pipeline for AIC 2026 Video Retrieval: dataset audit, CLIP retrieval baseline, temporal localization, VQA, TRAKE alignment, ranking, and evaluation.

## Current milestone: Phase 0 — Dataset Audit + Integrity Manifest

The repository intentionally does **not** contain competition data. Put the downloaded ZIP archives outside Git and point the audit script at the extracted dataset root.

Expected components:

```text
data/
├── keyframes/
├── clip/
├── mapping/
├── media_info/
└── objects/
```

The current dataset audit reports 177,321 keyframes across 873 videos, with 873 CLIP feature files, 873 mapping CSVs, 873 media-info JSONs, and 177,321 object JSONs.

### Phase 0 outputs

Run:

```bash
python -m venv .venv
# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

python -m aic2026.audit --config configs/example.yaml
python -m aic2026.validate --manifest artifacts/dataset_manifest.parquet
```

The audit produces:

```text
artifacts/
├── dataset_manifest.csv
├── dataset_manifest.parquet
├── dataset_audit.json
└── audit_report.txt
```

The manifest joins keyframe images, original video `frame_id`, `pts_time`, `fps`, CLIP row, and object JSON path. Integrity checks flag missing per-video CLIP/mapping files and row-count mismatches before retrieval work begins.

Competition data should not be committed to Git. See `.gitignore`.

## Roadmap

1. Phase 0 — dataset integrity and unified manifest
2. Phase 1 — CLIP retrieval baseline + AIC ranking metrics
3. Phase 2 — multimodal reranking with objects/metadata
4. Phase 3 — coarse-to-fine temporal localization
5. Phase 4 — TRAKE semantic keyframe alignment
6. Phase 5 — Q&A/VLM answer extraction
7. Phase 6 — Top-100 ranking optimization and submission pipeline
