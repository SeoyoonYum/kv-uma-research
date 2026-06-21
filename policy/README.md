# policy/ — Phase 2 (after Go)

Adaptive keep / recompute / evict policy on the mlx-lm KVCache path, driven by the
measured UMA cost model (no PCIe term; bandwidth-contention + compute terms).

Empty until Phase 1 clears the Go/No-Go gate (RESEARCH.md §8). Baselines to beat:
rotating, LRU, full-keep, always-recompute, with the Exp3 oracle as the upper bound.
