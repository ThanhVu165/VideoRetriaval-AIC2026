# AIC 2025 reference analysis

Reference repository: `lducc/hcm-aic`.

The reference implementation describes a local multimodal search pipeline using a common `frames.csv` catalog plus multiple retrieval channels: OpenAI CLIP ViT-B/32, optional BEiT-3 Large 384, ASR, OCR, and BLIP captions. It fuses channel results with reciprocal-rank fusion (RRF), then applies optional verification, neighborhood reranking, local within-video refinement, temporal-chain refinement, and deduplication.

## Relevant ideas to port to AIC2026

1. **Common frame catalog as the contract.** The reference keeps a small catalog that maps FAISS row IDs to video IDs, frame IDs, timestamps, and frame paths. Our `clip_frames_manifest.parquet` plays the same role and already has the necessary row alignment.

2. **Multi-channel retrieval instead of one CLIP score.** The reference can combine visual CLIP/BEiT-3 with ASR, OCR and captions using RRF. We should preserve this as an extensible late-fusion stage rather than forcing all evidence into one embedding.

3. **Query routing/decomposition.** The reference uses a typed retrieval plan with one to three English visual descriptions, optional ASR/OCR/caption queries, structured constraints, and optional ordered temporal events. This directly supports the multi-evidence direction planned for AIC2026.

4. **Shortlist then localize.** Expensive local evidence is collected only for a small number of candidate videos. This matches our coarse retrieval -> fine temporal localization design.

5. **Within-video neighborhood refinement.** Nearby keyframes in the same video are re-scored, while local ASR/OCR evidence can be attached to a temporal window.

6. **Temporal-chain scoring for ordered events.** For multi-event queries, the reference scores each event on nearby frames and searches for a chronological same-video chain within a bounded span. This is directly relevant to TRAKE.

7. **Ground-truth benchmark format.** The reference benchmark accepts query labels with `query_id`, `query`, `video_id`, `start_sec`, and `end_sec`, and reports video/moment ranks at multiple stages. This is a stronger evaluation contract than our current empty GT template and should inform the AIC2026 evaluator design.

8. **Trace every ranking stage.** The reference records raw RRF, verification, neighbor, local-refine, temporal-chain, and dedup stages. We should add similar per-query diagnostics so every ranking improvement is measurable.

## Important differences / caution

- The reference was built for AIC2025 and uses organizer/release artifacts with channels such as ASR, OCR and captions. We should only enable channels for which AIC2026 data actually exists and whose row/frame alignment is verified.
- The reference uses Gemini only as a query router. It does not require Gemini to produce the final retrieval answer. This pattern is useful, but external API use should remain optional and deterministic ablations should remain available.
- The AIC2026 dataset contract and official scoring rules remain authoritative. This document records transferable engineering ideas, not official AIC2026 rules.
