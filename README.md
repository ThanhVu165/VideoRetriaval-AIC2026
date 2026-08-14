# VideoRetrieval-AIC2026

Pipeline for AIC 2026 Video Retrieval: dataset audit, CLIP retrieval baseline, temporal localization, VQA, TRAKE alignment, ranking, and evaluation.

## Current milestone: Phase 0 — Dataset Audit

The repository intentionally does **not** contain competition data. Put the downloaded ZIP archives outside Git and point the audit script at the extracted dataset root.

Expected components:

```text
data/
├── keyframes/
├── clip/
├── mapping/
├── media_info/
└── objects/              # optional for Phase 0
```

See `docs/dataset_layout.md` for the expected layout and `configs/example.yaml` for configuration.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m aic2026.audit --config configs/example.yaml
```

The audit produces:

```text
artifacts/
├── dataset_manifest.csv
├── dataset_audit.json
└── audit_report.txt
```

Competition data should not be committed to Git. See `.gitignore`.
