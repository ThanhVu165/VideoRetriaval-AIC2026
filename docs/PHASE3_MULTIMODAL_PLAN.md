# Phase 3 — Multimodal Retrieval Fix

## Evidence from manual audit

The current CLIP-only retrieval baseline failed a cycling query: manual inspection showed the retrieved IDs were mostly unrelated while the local corpus contains a strong cluster of cycling videos under `L23_Vxxx`.

The current CLI can already accept caption/OCR/ASR evidence stores, and the pipeline already has object-path and metadata hooks, but a test run with no evidence arguments reported an empty evidence modality set. The immediate goal is to make available official/supporting evidence part of the default retrieval path instead of requiring manual flags for each run.

## Target architecture

Query → query decomposition → parallel candidate generation → rank-level fusion → video-level reranking → temporal localization → semantic keyframe.

Modalities:

- CLIP frame retrieval: primary visual candidate generator.
- Caption/ASR/OCR SQLite evidence: text evidence and timestamp/keyframe clues.
- Object JSON: object/entity evidence from the provided detection artifacts.
- Media metadata: video-level prior and semantic hints when present.
- Temporal localization: separate downstream stage; do not treat a static frame similarity as proof of a temporal proposition.

## Design constraints

1. Keep the existing CLIP/FAISS baseline intact as a regression baseline.
2. Never overwrite baseline artifacts; new experiments use separate output names.
3. Use rank-level fusion (RRF-style) across heterogeneous evidence instead of assuming comparable raw score scales.
4. Missing modalities must degrade gracefully with explicit diagnostics.
5. The output must report which modalities were available/used and the per-modality evidence for every candidate.
6. Official videos remain the source of truth. Supporting artifacts are evidence only.
7. No synthetic GT is introduced. Manual validation remains the current evaluation mechanism until official GT exists.

## First implementation target

Make `retrieve` auto-discover available caption/OCR/ASR evidence artifacts and configured media metadata/object roots from the repository layout. Add a `--multimodal` opt-in for the new path initially, preserving the current command behavior for baseline comparison.

For the first regression query, verify that cycling candidates under `L23_Vxxx` enter the candidate set and that the final diagnostic output can explain why a video was ranked.
