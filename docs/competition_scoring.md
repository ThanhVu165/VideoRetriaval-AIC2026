# AIC 2026 competition scoring contract

This document records only rules verified from the supplied **Thông tin vòng Sơ tuyển AIC 2026** PDF. It is not a submission-format specification.

## 1. Candidate limit and ranking cutoffs

For each query, the team may submit at most 100 answers. The official ranking cutoffs are:

```text
1, 5, 20, 50, 100
```

For each cutoff `k`, the organizer defines:

```text
R@k = max(R-Score(r_i)) for 1 <= i <= k
```

The query Final Score is:

```text
Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
```

Therefore the correct implementation is **not** binary recall over video IDs and is **not** MRR.

## 2. Textual KIS

Submission answer:

```text
<video_id>, <frame_id>
```

For ground-truth video `GT_v` and valid frame interval `[s,e]`:

```text
R-Score = I(video_id == GT_v AND s <= frame_id <= e)
```

## 3. Q&A

Submission answer:

```text
<video_id>, <frame_id>, <answer>
```

R-Score is 1 only when all three conditions hold:

```text
video_id == GT_v
s <= frame_id <= e
answer == GT_answer (semantic match)
```

The organizer describes answer correctness as semantic. This repository therefore exposes an injected `answer_matcher` instead of assuming that lowercase/whitespace normalization is the official semantic evaluator.

## 4. TRAKE

Submission answer:

```text
<video_id>, <frame_id_1>, ..., <frame_id_N>
```

There is a strict video gate:

```text
submitted video != GT video  -> R-Score = 0
```

When the video is correct, R-Score is the fraction of event frames that fall inside their corresponding ground-truth intervals:

```text
R-Score = number_of_matching_events / N
```

## 5. Engineering implication

`aic2026.competition_metrics` contains pure scoring primitives only. It deliberately does **not** define an official query-file parser, ground-truth file schema, or submission serializer because those formats must be taken from the actual BTC package when supplied.

Existing `aic2026.metrics.recall_at_k()` remains useful for retrieval diagnostics, but it must not be presented as the official AIC 2026 Final Score.
