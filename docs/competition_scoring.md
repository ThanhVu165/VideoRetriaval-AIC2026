# AIC 2026 competition scoring contract

This document records only rules verified from the supplied **Thông tin vòng Sơ tuyển AIC 2026** PDF. It is not a submission-format specification.

## 1. Candidate limit and ranking cutoffs

For each query, the team may submit at most 100 answers. The official ranking cutoffs are:

```text
1, 5, 20, 50, 100
```

For each cutoff `k`:

```text
R@k = max(R-Score(r_i)) for 1 <= i <= k
```

and:

```text
Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
```

This is not binary video recall and not MRR.

## 2. Textual KIS

For ground-truth video `GT_v` and valid frame interval `[s,e]`:

```text
R-Score = I(video_id == GT_v AND s <= frame_id <= e)
```

## 3. Q&A

R-Score is 1 only when all three conditions hold:

```text
video_id == GT_v
s <= frame_id <= e
answer is semantically correct
```

The answer matcher is injected because this repository must not invent the organizer's semantic-answer evaluator.

## 4. TRAKE

There is a strict video gate:

```text
submitted video != GT video  -> R-Score = 0
```

When the video is correct:

```text
R-Score = number_of_matching_events / N
```

where each submitted event frame is checked against its corresponding ground-truth interval.

## 5. Repository boundary

`aic2026.competition_metrics` contains pure scoring primitives only. It does not define an official query-file parser, ground-truth schema, or submission serializer. Those must be taken from the actual BTC package when available.

The development benchmark may use the repository's explicit local GT JSON template to exercise these primitives. That local schema must not be described as the organizer's official schema.
