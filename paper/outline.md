# Paper outline (seed)

Working title: KV-cache decision policy for single-pool unified memory.
Target: MLSys — workshop first (EuroMLSys / MLArchSys), then main track.

## Sections (draft)
1. Introduction — PCIe-era KV management; the unified-memory coordinate; why it breaks.
2. Background & related work — FlexGen / vLLM swap-vs-recompute; KV-Direct N≈50; the UMA spectrum.
3. Cost structure of single-pool UMA (Exp1) — keep-and-read vs drop-and-recompute curves.
4. Shared-bus contention (Exp2) — CPU/GPU bandwidth contention shifts N*.
5. Policy gap (Exp3) — current frameworks vs a measured-cost oracle.
6. (Phase 2) Adaptive policy — beats baselines, avoids OOM.
7. Cross-arch contrast (Exp4) — N* across the unified-memory spectrum.

## Claims to support with data
- The "offload to CPU" lever degenerates in single-pool UMA (the swap option vanishes).
- keep-vs-recompute N* differs from PCIe / hardware-agnostic N≈50 predictions.
- The discrepancy widens with model size.
