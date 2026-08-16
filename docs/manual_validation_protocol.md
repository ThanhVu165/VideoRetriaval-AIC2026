# AIC 2026 — Manual Validation Protocol

## Purpose

The supplied competition query workbook is available, but no organizer ground-truth package is available to the team. Therefore local evaluation must distinguish **retrieval generation** from **human validation**.

This document defines the validation contract without pretending that manually recorded labels are official AIC ground truth.

## Workflow

```text
Query Manifest
    ↓
System Retrieval
    ↓
Top-100 candidates
    ↓
Human inspection
    ↓
Local validation record
    ↓
Benchmark diagnostics
```

## What the validator records

For each query, record only facts that can be verified from the supplied videos / frames:

- `query_id`
- `task_type`
- `candidate_rank`
- `video_id`
- `frame_id` (or event-specific frame IDs for TRAKE)
- `video_match`: whether the candidate video contains the requested event
- `frame_match`: whether the submitted frame is semantically inside the manually verified event interval
- `answer_match`: for Q&A only, whether the generated answer is semantically correct
- `notes`: optional explanation for ambiguous cases

For TRAKE, record one validation item per event and preserve event order.

## Important status distinction

A manually validated record is a **local human-validation label**. It is not the organizer's official GT and must not be described as such.

The competition scorer may consume these labels only through an explicit adapter that declares the local contract.

## Recommended validation strategy

Do not inspect all 100 candidates for every query initially.

Start with:

- Top-1
- Top-5
- Top-20

If the correct candidate is absent, extend inspection to Top-50 and Top-100.

For temporal tasks, inspect the original video around the candidate frame rather than relying only on a keyframe thumbnail. Preserve the original video `frame_id`.

## Task-specific validation

### TKIS / Textual KIS

A candidate is locally correct when:

1. the video contains the described event; and
2. the candidate frame lies inside the manually verified event interval.

### Q&A

A candidate is locally correct only when:

1. the video is correct;
2. the frame is inside the manually verified event interval; and
3. the answer is semantically correct.

### TRAKE

A candidate is locally correct only when:

1. the selected video is correct; and
2. each event's selected semantic keyframe lies inside that event's manually verified interval.

Keep the event ordering from the query manifest.

## Why this is separate from the official evaluator

The official AIC evaluator and submission schema have not been supplied. This protocol therefore exists to create a reproducible internal benchmark, not to claim exact official scoring semantics beyond the scoring rules already verified from the BTC documentation.

## Engineering rule

Every model/ranking change should be evaluated against the **same manually validated query set**. Never compare two systems using different validation subsets.
