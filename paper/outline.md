# Paper outline — A Measurement Study of Unified-Memory LLM Inference under CPU/GPU Contention

Working title: **"When CPU Traffic Changes KV-Cache Decisions: A Measurement Study of Unified-Memory LLM Inference"**
Venue: **ML for Systems @ NeurIPS** (4-page extended abstract, non-archival → arXiv + main resubmit).
Phase 1 = a complete measurement paper.

NEW thesis (topic pivot 2026-06-23 #2): In single-pool UMA, **external CPU memory traffic creates an
asymmetric interference channel** for LLM inference — decode is bandwidth-arbitration sensitive while
prefill/recompute is comparatively compute-bound. This asymmetry **changes the cost model for KV-cache
management** and makes static PCIe-era policies insufficient.
Arc: **cost structure (setup) → contention asymmetry (MAIN) → it changes KV decisions → simple policy isn't enough → prediction/contention-aware needed.**

> **TOPIC PIVOT (2026-06-23 #2, see RESEARCH §4).** Now a *measurement study* with **Finding 2 (CPU/GPU
> contention asymmetry) as the paper's subject**, not a KV-policy paper. → **Sections below need reordering
> on writing:** §5 (contention) becomes the spine/main; §3–4 (cost structure + crossover) demote to *setup/background*;
> §6–7 (policy gap + Phase 2a) become the *KV-decision-implication + design-lesson* tail. Honest KV link: contention
> *shifts* the keep-vs-recompute boundary (write as "must be modeled", not "fully reverses"); back it with the
> planned α-sweep + contention A/B (EXPERIMENTS [P1]). The figure manifest stays valid; only section emphasis/order moves.
>
> **Framing re-centered (2026-06-23, see RESEARCH §4/§4b).** Novelty center of mass moved:
> Finding 1 (eviction = forced recompute, 600–1,600×) → **background/premise** (Agent Memory
> Below the Prompt, 2603.04428, is a published *ally* that co-establishes it; our delta = controlled
> measurement + model-size scaling 329→1,587× only). **Finding 2 (CPU↔GPU single-UMA-bus contention
> asymmetry + KV-decision implication) = the center novelty** (empty after 3 novelty searches; the
> novelty is the *combination* — external non-LLM CPU traffic × single pool × KV decision — not
> "asymmetry" itself, which is red-ocean vs Nexus/DuetServe GPU-internal prefill↔decode). Finding 3
> (prediction needed) → **motivation/future**, must cite datacenter priors (SAECache 2605.18825 etc.).
> Realism defense for Finding 2: on-device LLMs share memory with agent tools/RAG/OS/preprocessing →
> shared single-device use is UMA's *default operating mode*, so contention is not a corner case.
> Prose stays in the strategy chat; section text below is the pre-re-centering scaffold — update on writing.

All numbers below are measured on M4 MacBook Air (16 GB, fanless), MLX/mlx-lm 0.31, Qwen2.5-4bit.
Writing happens in the strategy chat; this file is the scaffold + figure/result manifest.

## Sections

1. **Introduction** — PCIe-era KV management assumes tiered transfer (GPU↔CPU over a slow bus).
   Single-pool UMA (Apple Silicon, and the consumer end of the GH200/GB200 spectrum) collapses
   the "offload to CPU" tier → the decision reduces to keep-and-read vs drop-and-recompute, and a
   *new* axis appears (CPU/GPU shared-bus contention). Contributions ↓.

2. **Background & related** — FlexGen LP / vLLM swap-vs-recompute (PCIe-tiered); KV-Direct N≈50
   (hardware-agnostic); the UMA spectrum (Apple single-pool ↔ NVIDIA C2C two-pool); vLLM-on-Mac
   (vllm-metal / vllm-mlx — PagedAttention *mechanism* ported, NOT cost-remodeled; eviction still
   LRU/recompute-default); folklore (backend.ai / touchdown-labs / min.io blogs) → we give the
   first controlled/mechanistic measurement + cost model. (RELATED_WORK.md)

3. **Cost structure of single-pool UMA (Exp1)** — prefill_ms(N) and decode_step_ms(N), N∈[128..8192],
   0.5/1.5/3/7B. decode floor ≈ per-step *weight read* (≈ weights / 120 GB·s, validated: 1.5B floor
   ~11 ms ↔ ~1.3 GB read); the N-dependent KV-read term runs at ~62 GB/s *effective* (≈ half the
   120 GB/s peak). prefill is super-linear (O(N²) attention emerges past N≈1024).
   → fig: **exp1_costcurve_1.5B_20260620.png** (prefill & decode vs N). data: results/csv/exp1_1.5B_*.csv

4. **The crossover and its scaling (the "gap")** — keep-and-read dominates drop-and-recompute at
   *every* N (crossover bandwidth B* ≈ 0.02–0.04 GB/s ≪ any real interconnect). The UMA-vs-PCIe
   difference lives in the **eviction path**: PCIe swaps KV to CPU DRAM (~19 ms, preserved); UMA
   has no fast second tier → must drop+recompute (~12 s) ≈ **600×** more @N=8192 (1.5B), growing
   with model size to **~1,587×** (7B@4096). recompute "tax" = prefill/decode = 11→785 decode-steps.
   → figs: **exp1_crossover_1.5B_20260620.png**, **exp1_size_compare_20260620.png**.
   ⚠ caveat (methodology contribution): fanless thermal inflation on large-N absolutes — 7B@8192
   prefill throttles (powermetrics: GPU 854–997 vs 1465 MHz); clean 7B@4096 = 20.9 s vs cumulative
   sweep 29.8 s (42% inflation). Relative results (recovery%, crossover direction) are thermal-robust.

5. **Shared-bus contention asymmetry (Exp2) — the cleanest novelty.** A concurrent CPU memory-
   bandwidth load (native STREAM, P-core biased) slows GPU **decode up to ~38%** at ~69 GB/s load.
   Mechanism nailed by a **powermetrics A/B**: under load the GPU clock stays maxed / rises and
   power rises while throughput falls → **bandwidth/arbitration contention, NOT thermal throttling.**
   **prefill (recompute) is nearly immune** (−0.5 ~ +7.5 %; compute-bound) → contention shifts N*
   *toward recompute* (~1.3–1.4×): a UMA-specific lever absent from PCIe models. Size-robust
   (~38–44 % across 0.5–7B → a structural effect, no per-model tuning).
   → figs: **exp2_contention_1.5B_20260621.png** (+ prefill-vs-decode asymmetry; size sweep
   exp2_{0.5,3,7B}_20260622.csv). data: results/csv/exp2_*.csv, exp2pf_*, exp2pm_*

6. **Policy gap (Exp3, cost-model simulation).** Replay real ShareGPT multiturn under
   rotating / full_keep / always_recompute / LRU / causal / oracle, scored by the measured cost
   model. On real traces: rotating (mlx-lm default) silently drops ~69 % of context; full_keep OOMs;
   always_recompute over-pays; **oracle (foresight) ≫ LRU**. A simple **causal** heuristic
   (drop_gain = keep_ms − P_reuse·prefill(N), recency-only, no foresight) recovers **41–83 %** of the
   LRU→oracle gap in simulation, robust across budget×concurrency (50–92 %).
   → fig: **exp3_policygap_1.5B_20260622.png**. data: results/csv/exp3_1.5B_*.csv

7. **From simulation to the real engine — the simple-policy limit (Phase 2a).** We implement causal
   in the real mlx-lm engine (subclass of `LRUPromptCache`, eviction victim = drop_gain;
   policy/causal_cache.py). On the real 1.5B model under memory pressure, **causal does NOT beat LRU**
   at any K (worse, and worse as K rises). Diagnosis: keep_ms (~0–2 ms) ≪ recompute (100s–1000s ms),
   so drop_gain degenerates to "evict the cheapest-recompute (small) sequence" → tiny memory freed
   per eviction → **eviction churn** → more recomputes; pure recency (LRU) is more memory-efficient.
   The sim's small causal-over-LRU edge (~2–4 % absolute) doesn't survive real-engine dynamics.
   → fig: **exp_phase2a_causal_vs_lru.png** (TTFT/recomputes: LRU baseline vs causal across K).

8. **Discussion & future work.** Contributions = (i) re-derivation of the single-pool decision space
   (offload degenerates → keep-vs-recompute), (ii) first controlled/mechanistic measurement of the
   eviction-cost collapse (600–1600×, model-scaling), (iii) first measurement + mechanism of CPU/GPU
   shared-bus contention asymmetry (a new N*-moving axis). Three findings from Phase 2a:
   (a) the limit of a simple recency+recompute heuristic, (b) a sim↔real gap (cost-model sims
   over-estimate policy wins), (c) **reuse PREDICTION is essential** (only the foresight oracle beats
   LRU). Future work: a predictive UMA-native policy; a real serving-engine port (Phase 2b / vllm-metal,
   lab-conditional); fan-equipped large-N cross-check; cross-arch N* (Exp4, PCIe GPU).

## Claims ↔ evidence
- Offload lever degenerates (swap → forced recompute, ~600–1,587×, grows with size). — Exp1 + crossover + size sweep.
- keep-vs-recompute N* differs from PCIe / N≈50, widens with model size. — crossover + size sweep.
- New axis: CPU/GPU shared-bus contention is asymmetric (decode −38%, prefill immune; bandwidth not thermal; size-robust). — Exp2 + powermetrics A/B + prefill-contention + size sweep.
- Current frameworks leave a policy gap (oracle ≫ rotating/LRU; rotating drops 69% context). — Exp3 sim.
- A simple recency+recompute policy does NOT close it on the real engine → prediction is the lever. — Phase 2a (honest negative).

## Figure manifest (results/figures/)
| paper § | figure | shows |
|---|---|---|
| 3 | exp1_costcurve_1.5B_20260620.png | prefill (super-linear) & decode (floor + KV slope) vs N |
| 4 | exp1_crossover_1.5B_20260620.png | recompute vs read-back recovery cost (recompute ~1000× over read) |
| 4 | exp1_size_compare_20260620.png | UMA-vs-PCIe gap 329→1587× with model size; decode floor ∝ weights |
| 5 | exp2_contention_1.5B_20260621.png | decode tok/s & slowdown vs CPU bandwidth load |
| 5 | (asymmetry / size: exp2pf_*, exp2_{0.5,3,7B}_*) | prefill-immune vs decode-hit; ~40% size-robust |
| 6 | exp3_policygap_1.5B_20260622.png | per-policy metrics + causal recovery of LRU→oracle gap |
| 7 | exp_phase2a_causal_vs_lru.png | real model: causal worse than LRU at every K (the limit) |
