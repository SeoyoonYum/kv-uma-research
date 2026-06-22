# policy/ — Phase 2 (after Go)

Adaptive keep / recompute / evict policy on the mlx-lm KVCache path, driven by the
measured UMA cost model (no PCIe term; bandwidth-contention + compute terms).
Go/No-Go cleared (Phase 1 GO). Baselines to beat: rotating, LRU, full-keep,
always-recompute, with the Exp3 oracle as the upper bound. Causal heuristic validated in
sim (Exp3: recovers 50–92% of the LRU→oracle TTFT gap on real ShareGPT).

## Phase 2a hook-point recon (mlx-lm 0.31.3, `mlx_lm/models/cache.py`)

The sequence-level prompt cache is **`LRUPromptCache(max_size, max_bytes)`** — exactly the
"current engine's recency-based eviction" our baseline represents (it IS LRU here).

- `PromptTrie` does prefix matching = multiturn KV reuse (`fetch_nearest_cache`).
- Each cached sequence is a `CacheEntry(prompt_cache, nbytes, cache_type)`.
- **Eviction victim selection = `self._lru.pop()`** (the `CacheOrder` deque, least-recently-
  used), called from `insert_cache` (over `max_size`/`max_bytes`) and `trim_to(n_sequences, n_bytes)`.

**Causal hook (Phase 2a):** subclass `LRUPromptCache` → override victim selection to pick by
`drop_gain = keep_ms − P_reuse·recompute_ms` instead of pure LRU. Every input is present at
the eviction site:
- `keep_ms`       = `entry.nbytes` / ~92 GB·s  (memory freed, as read-time)
- `recompute_ms`  = Exp1 `prefill_ms(len(tokens))` curve lookup  (superlinear in N)
- `P_reuse`       = `exp(−idle/K)` from the entry's LRU recency
(Same `drop_gain` as the Exp3 sim; oracle is offline, causal is recency-only.)

**Measure (Phase 2a):** run real multiturn (prefix reuse via `make_prompt_cache`) under
rotating / full_keep / always_recompute / LRU(stock) / causal; report TTFT, inter-token
latency, peak memory, throughput, OOM; compare to the Exp3 sim recovery%.

Other relevant API: `KVCache.trim(n)` / `trim_prompt_cache()` (token-level trim),
`RotatingKVCache(max_size, keep)` (mlx-lm default windowed cache = our `rotating` baseline),
`can_trim_prompt_cache()`, `entry.nbytes` / cache `.size`/`.offset`.

**off-ramp:** if real numbers diverge hard from the sim or hit the 16 GB wall, record that
as a finding (sim↔real gap / memory limit) and ship Phase 1 for the workshop. Don't wall-bang.
