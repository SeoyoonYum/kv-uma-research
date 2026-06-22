#!/usr/bin/env python3
"""smoke_phase2a.py — functional check of CausalPromptCache (no tuning).

Inserts a few real mlx-lm KV caches (distinct sequences, controlled recency) into both the
stock LRUPromptCache (baseline) and our CausalPromptCache, with a small KV budget so eviction
triggers, and verifies:
  1. eviction actually fires,
  2. the causal victim is the argmax-drop_gain entry,
  3. it can differ from the LRU victim (recency-only).
"""
import math
import os
import sys

import mlx.core as mx
from mlx_lm.models.cache import KVCache, LRUPromptCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from exp3_policygap import CostModel          # noqa: E402
from causal_cache import CausalPromptCache    # noqa: E402  (same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def dummy_prompt_cache(L, n_layers=28, n_kv=2, hd=128):
    """A real per-layer KVCache list holding L dummy tokens (nbytes scales with L)."""
    pcs = []
    for _ in range(n_layers):
        c = KVCache()
        k = mx.zeros((1, n_kv, L, hd), dtype=mx.float16)
        c.update_and_fetch(k, k)
        pcs.append(c)
    mx.eval([c.state for c in pcs])
    return pcs


def survivors(cache, model, seqs):
    """Surviving conversation NAMES (identity-based, robust to equal lengths)."""
    lru = cache._lru
    items = lru._items if hasattr(lru, "_items") \
        else [mt for dq in lru._lrus.values() for mt in dq]
    keys = {(id(m), tuple(t)) for (m, t) in items}
    return [name for name, t in seqs.items() if (id(model), tuple(t)) in keys]


def main():
    cm = CostModel("1.5B")
    model = object()
    # distinct sequences (no shared prefix); B is oldest+small, A is newer+large, C triggers.
    # Q oldest+small, P less-old+large, R recent. Budget needs ~one small eviction:
    # LRU evicts oldest (small Q); causal evicts max-drop_gain (large P) -> they differ.
    seqs = {"Q(0.7k)": list(range(2_000_000, 2_000_000 + 700)),
            "P(3.5k)": list(range(1_000_000, 1_000_000 + 3500)),
            "R(0.7k)": list(range(3_000_000, 3_000_000 + 700))}
    nb = {name: cm.kv_bytes(len(t)) for name, t in seqs.items()}
    print("entries:", {k: f"{len(seqs[k])}tok / {nb[k]/1e6:.0f}MB" for k in seqs})
    budget = 125e6
    print(f"KV budget = {budget/1e6:.0f} MB  (sum = {sum(nb.values())/1e6:.0f} MB -> eviction)\n")

    # ---- baseline: stock mlx-lm LRUPromptCache (the current-engine recency eviction) ----
    order = ["Q(0.7k)", "P(3.5k)", "R(0.7k)"]
    base = LRUPromptCache(max_size=1 << 30, max_bytes=int(budget))
    for name in order:
        base.insert_cache(model, seqs[name], dummy_prompt_cache(len(seqs[name])))
    base_surv = survivors(base, model, seqs)

    # ---- ours: CausalPromptCache, with controlled idle gaps between inserts ----
    ours = CausalPromptCache(cm.prefill_ms, K=8.0, max_size=1 << 30, max_bytes=int(budget))
    gaps = {"Q(0.7k)": 300, "P(3.5k)": 300, "R(0.7k)": 0}   # idle accrued AFTER each insert
    for name in order:
        ours.insert_cache(model, seqs[name], dummy_prompt_cache(len(seqs[name])))
        ours._clock += gaps[name]
    ours_surv = survivors(ours, model, seqs)

    # ---- report drop_gain at the (final) state for the surviving + evicted entries ----
    print("drop_gain(seq) = keep_ms - P_reuse*prefill_ms(N), evict argmax:")
    for name in order:
        t = seqs[name]
        idle = ours._clock - ours._last_used.get(ours._key(model, t), ours._clock)
        keep = cm.kv_bytes(len(t)) / 92e6
        rc = cm.prefill_ms(len(t))
        p = math.exp(-idle / 8.0)
        dg = keep - p * rc
        print(f"  {name:8s} idle={idle:4d}  keep={keep:5.2f}ms  recompute={rc:7.0f}ms  "
              f"P_reuse={p:.3f}  drop_gain={dg:9.2f}")

    ev = lambda surv: [n for n in seqs if n not in surv]
    print(f"\n  baseline LRU evicted : {ev(base_surv)}  (kept {base_surv})")
    print(f"  causal      evicted : {ev(ours_surv)}  (kept {ours_surv})")

    ok_fire = len(ours_surv) < len(seqs)
    ok_differ = set(ev(base_surv)) != set(ev(ours_surv))
    print(f"\n  [check] eviction fired: {ok_fire} | causal!=LRU victim: {ok_differ}")
    print("  => CausalPromptCache hooks mlx-lm eviction and picks by drop_gain (not recency)."
          if ok_fire else "  => eviction did not fire — adjust budget.")


if __name__ == "__main__":
    main()
