# AIC 2026 — Project Context / Frozen Direction

> **Canonical engineering context for AI agents working on this repository. Read this before changing architecture or adding major components.**
>
> Last frozen: 2026-08-16.

## 1. Mission

Build a competition-ready Video Retrieval & Video Understanding system for AIC 2026 covering:

1. **Textual Known Item Search (Textual KIS)** — correct video + frame for a natural-language event description.
2. **Q&A** — correct video + frame + answer.
3. **TRAKE** — correct video + ordered semantic keyframe for each event in a sequence.

Optimize the competition ranking, not merely one plausible answer.

## 2. Frozen architectural decision

**Do NOT rewrite or replace the current repository. Extend it incrementally.**

Preserve and build on:

```text
Dataset audit
    -> unified manifest
    -> CLIP frame index / retrieval
    -> candidate video aggregation
    -> multimodal reranking hooks
    -> temporal localization skeleton
    -> ranking / Top-K hooks
```

`lducc/hcm-aic` is a **reference implementation/source of ideas**, not the base to fork or replace this project with.

Use it selectively for ideas such as multi-channel retrieval, query decomposition, OCR/ASR retrieval, and score fusion. Adopt a component only after checking its actual implementation and measuring it against our local evaluator.

## 3. Current verified state

The repository has a real Phase-0/Phase-1 foundation:

- dataset audit and validation;
- unified manifest preserving original video `frame_id` separately from keyframe ordinal;
- CLIP frame retrieval using normalized embeddings;
- optional FAISS backend;
- video aggregation using `max` or `topk_mean`;
- pipeline skeleton for multimodal reranking and temporal localization;
- ranking / Top-K hooks.

Current full dataset audit available to the project reports:

- 177,321 keyframes;
- 873 videos;
- 873 CLIP feature files;
- 873 mapping CSVs;
- 873 media-info JSONs;
- 177,321 object JSONs.

These are facts from current project sources. Do not silently substitute another dataset layout or invent missing query/ground-truth details.

## 4. Explicitly NOT complete yet

Do not describe the repository as competition-complete until these are implemented and tested:

- official AIC evaluator based strictly on supplied query / ground-truth / submission specification;
- Textual KIS end-to-end query encoding + retrieval + temporal localization + submission;
- fine-grained temporal localization robust to events whose correct interval can be under 10 frames;
- TRAKE multi-event semantic keyframe alignment and output;
- Q&A answer extraction/generation and answer-aware ranking;
- exact Top-100 submission generation;
- R@1, R@5, R@20, R@50, R@100 and Final Score evaluation;
- local benchmark/regression tests proving changes improve the competition objective.

A class/function/README entry is not evidence that the corresponding competition requirement is implemented. Trace the executable path and output contract.

## 5. Non-negotiable competition constraints

Use the official AIC/BTC specification in the project sources as authority. In particular:

- three task families: Textual KIS, Q&A, TRAKE;
- output may contain up to 100 candidates/answers as specified by the task;
- ranking matters at Top-1/5/20/50/100;
- Final Score is derived from those recall levels;
- TRAKE requires the correct video and ordered semantic keyframes for the event sequence;
- TRAKE temporal alignment is fine-grained; the correct region may be fewer than 10 frames;
- video is the official competition data; keyframes, object detections, CLIP features and metadata are auxiliary data.

If an exact evaluator/submission detail is not present in repository sources, inspect the official supplied specification first. **Do not invent a format.**

## 6. Frozen implementation order

Do not jump directly to large VLM/OCR/ASR/model additions. Work in this dependency order.

### Phase A — Evaluation foundation

Create the official evaluator first:

```text
aic2026/evaluator/
    kis.py
    qa.py
    trake.py
    final_score.py
```

It must measure the official task metrics and Final Score from the supplied specification.

### Phase B — Textual KIS

Complete the existing path:

```text
natural-language query
    -> official query representation / encoder
    -> candidate frame retrieval
    -> candidate video aggregation
    -> fine reranking
    -> temporal localization
    -> semantic keyframe
    -> Top-100 ranked output
```

Do not assume an unofficial query-file format or text encoder before inspecting the official query package.

### Phase C — Fine temporal localization

Upgrade the existing coarse-to-fine skeleton so local decoding/scoring does not lose very short events. The current coarse-window logic is a starting point, not the final TRAKE-grade localizer.

Target structure:

```text
coarse candidate
    -> dense local decoding
    -> fine frame/event scoring
    -> short temporal window
    -> semantic peak/keyframe
```

### Phase D — TRAKE

Implement TRAKE as a first-class multi-event pipeline:

```text
TRAKE query
    -> event decomposition / event list
    -> candidate video retrieval
    -> per-event temporal localization
    -> one semantic keyframe per event
    -> ordered event/keyframe output
    -> ranking/evaluation
```

Do not force multi-event TRAKE semantics into the existing single-event `LocalizedEvent` abstraction if that obscures event ordering or independent alignment.

### Phase E — Multimodal retrieval

Only after evaluation + baseline retrieval + temporal benchmark are measurable, add/finalize signals such as:

```text
visual / CLIP
objects
metadata
OCR (if supported by actual data/tooling)
ASR (if supported by actual data/tooling)
other approved multimodal representations
```

Fuse them through measurable reranking/score-fusion experiments.

`hcm-aic` may be consulted for multi-channel retrieval and fusion patterns, but its architecture/documentation is not proof that a component is implemented here.

### Phase F — Q&A

Complete:

```text
question
    -> video retrieval
    -> temporal localization
    -> relevant frame/clip
    -> answer extraction/generation
    -> answer normalization
    -> answer-aware ranking
    -> video + frame + answer output
```

### Phase G — Ranking optimization

Optimize the actual competition objective:

```text
R@1
R@5
R@20
R@50
R@100
Final Score
```

Do not optimize only one recall level or only model similarity.

### Phase H — Submission / regression

Build a deterministic submission pipeline and regression suite. Every major change should be evaluated against the same local benchmark before acceptance.

## 7. Engineering rules for future AI agents

1. **Read this file before major changes.**
2. **Do not reset the architecture without evidence of structural failure.**
3. **Do not claim a requirement is implemented merely because a class/function/README entry exists.** Trace the executable path and output contract.
4. **Do not invent official query, ground-truth, scoring, or submission formats.** Inspect supplied sources first.
5. **Prefer measurable incremental changes over speculative model stacking.**
6. **Keep original frame IDs and temporal mappings exact.** Never confuse keyframe ordinal with original video frame ID.
7. **Preserve reproducibility.** New retrieval/ranking components should have deterministic configuration and benchmarkable outputs where practical.
8. **Do not commit competition data, large embeddings, archives, or generated artifacts unless explicitly required.**
9. **When using `hcm-aic`, separate reference-derived facts from our own implementation decisions.**
10. **Before declaring the project ready, verify all three tasks and all required ranking levels through the evaluator.**

## 8. Current priority

> **The next implementation milestone is the official evaluator + local benchmark.**

Do not start with OCR/ASR/VLM expansion until the evaluator and temporal benchmark are in place, unless a concrete dependency requires otherwise.

## 9. Definition of done

The system is competition-ready only when there is an executable path for all three tasks:

```text
Textual KIS -> video + semantic frame -> ranked Top-100
Q&A         -> video + frame + answer -> ranked Top-100
TRAKE       -> video + ordered semantic keyframes -> ranked Top-100
```

and the repository can evaluate those outputs using the official scoring specification, including R@1/R@5/R@20/R@50/R@100 and Final Score.
