# AIC 2026 — Project Context / Frozen Direction

> **Canonical engineering context for AI agents. Read this before major changes.**
>
> Frozen: 2026-08-16.

## Mission

Build a competition-ready Video Retrieval + Video Understanding system for AIC 2026 covering:

1. **Textual KIS** → `video_id + frame_id`
2. **Q&A** → `video_id + frame_id + answer`
3. **TRAKE** → `video_id + ordered semantic keyframes`

Optimize the actual competition ranking, not only single-best retrieval.

## Frozen architecture

**Do not rewrite or replace the current repository. Extend it incrementally.**

Current foundation:

```text
Dataset audit / unified manifest
        ↓
CLIP frame index + query encoder adapters
        ↓
Candidate video generation
        ↓
Multimodal reranking hooks
        ↓
Original-video temporal decoding
        ↓
Fine frame scoring / semantic peak
        ↓
TRAKE monotonic alignment / VQA adapter
        ↓
Ranking / Top-k
```

`lducc/hcm-aic` is a **reference**, not the base to fork. Use it selectively for ideas such as multi-channel retrieval, query decomposition, local refinement and temporal-chain ranking. Verify actual implementation before claiming a component exists here.

## Current development state

`phase3-completion-clean` extends the phase-2 foundation with:

- fine original-frame CLIP temporal scoring;
- an end-to-end TRAKE baseline with cross-event video selection and monotonic alignment;
- an end-to-end VQA runner with an optional Hugging Face VLM backend;
- CLI entry points for `retrieve --fine-score`, `trake`, and `vqa`;
- existing benchmark and ground-truth template utilities.

These are **baselines**, not evidence of official competition completeness. In particular, the official query/ground-truth/submission binding is still pending.

Do not reimplement existing modules blindly. Inspect the executable code and tests first.

## Dataset facts

The latest supplied full audit reports:

- 177,321 keyframes;
- 873 videos;
- 873 CLIP feature files;
- 873 mapping CSVs;
- 873 media-info JSONs;
- 177,321 object JSONs.

Video is the official competition data; keyframes, objects, CLIP features and metadata are supporting data. Preserve the mapping between keyframe ordinal and original video `frame_id` exactly.

## Official scoring facts already verified from the supplied BTC PDF

For every query, at most 100 answers are submitted. The official cutoffs are:

```text
R@1, R@5, R@20, R@50, R@100
```

For each cutoff:

```text
R@k = max R-Score among the first k submitted answers
```

and:

```text
Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
```

Task-specific R-Score:

### Textual KIS

```text
R = 1 iff submitted video == GT video
       AND submitted frame is inside [start, end]
```

### Q&A

```text
R = 1 iff video is correct
       AND frame is inside [start, end]
       AND answer is semantically correct
```

Semantic answer matching is intentionally exposed as an injected matcher; do not invent a local normalization rule and call it official.

### TRAKE

```text
wrong video -> R = 0
correct video -> R = (# event frames inside their GT intervals) / N
```

## Official scoring vs local diagnostics

`aic2026.metrics.recall_at_k()` is a generic retrieval diagnostic. It is **not** the official AIC2026 Final Score.

`aic2026.competition_metrics` contains pure competition scoring primitives verified from the BTC PDF. It must not invent an official query/GT/submission file schema.

The current benchmark's local GT JSON is only a development contract. It may be used to exercise the scoring primitives, but must not be described as the organizer's official schema.

## Frozen implementation order

Do not jump directly to OCR/ASR/VLM/model stacking.

```text
A. Competition scoring foundation
        ↓
B. Textual KIS end-to-end benchmark
        ↓
C. Fine temporal localization benchmark
        ↓
D. TRAKE multi-event evaluation/alignment
        ↓
E. Multimodal retrieval improvements
        ↓
F. Q&A answer generation/evaluation
        ↓
G. Ranking / Final Score optimization
        ↓
H. Deterministic submission + regression suite
```

At every step:

```text
Implement → Benchmark → Error analysis → Keep only measurable improvements
```

## Non-negotiable rules for future AI agents

1. Read this file before major architectural changes.
2. Do not reset the architecture without evidence of structural failure.
3. Trace executable code and output contracts; a README/class/function is not proof that a competition requirement is complete.
4. Never invent official query, ground-truth or submission formats.
5. Do not confuse keyframe ordinal with original video `frame_id`.
6. Do not treat auxiliary data as a replacement for official source videos.
7. Do not add a model merely because it sounds stronger; measure its effect on the competition objective.
8. Keep competition data, large embeddings and generated artifacts out of Git.
9. Separate facts derived from `hcm-aic` from facts about this repository.
10. Before declaring competition readiness, verify all three task families and all five ranking cutoffs through the competition evaluator.

## Definition of done

The repository is competition-ready only when there is an executable, tested path for:

```text
Textual KIS → ranked video + valid semantic frame
Q&A         → ranked video + valid frame + semantic answer
TRAKE       → ranked video + ordered event keyframes
```

and the outputs can be evaluated using the verified AIC2026 R-Score / R@1 / R@5 / R@20 / R@50 / R@100 / Final Score rules.