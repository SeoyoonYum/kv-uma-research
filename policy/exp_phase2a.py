#!/usr/bin/env python3
"""
exp_phase2a.py — Phase 2a: run the causal policy in the REAL mlx-lm cache on a real model.

Replays filtered ShareGPT multiturn traces through Qwen2.5-1.5B-4bit under a shared
sequence-level prompt cache with a small KV budget, comparing eviction policies:
  baseline = stock mlx-lm LRUPromptCache (the current-engine recency eviction)
  causal   = CausalPromptCache (drop_gain victim; Exp3 formula, recency-only)
Each turn fetches the longest cached prefix (prefix reuse), prefills only the uncached
remainder (TTFT), decodes the response, then re-inserts the cache (eviction fires here).
Measures mean/p95 TTFT, recompute count, peak memory — the real analogue of the Exp3 sim.

Controls (16 GB, fanless): single 1.5B run, mx.eval-forced, small KV budget for controlled
pressure (NOT the 16 GB wall), cooldown between policies. Token VALUES are dummy (cost is
shape-only, Exp1); per-conversation sequences are distinct-prefixed so the trie reuses correctly.
"""
import argparse
import os
import sys
import time

import numpy as np
import mlx.core as mx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import models, measure, workloads     # noqa: E402
from exp3_policygap import CostModel, build_schedule  # noqa: E402
from causal_cache import CausalPromptCache         # noqa: E402
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache  # noqa: E402

_PAT = 50000  # dummy token base (valid < vocab); position-patterned, conv distinct at pos 0
_MKEY = "model"  # hashable namespace key for the prompt-cache trie (the mlx Model is unhashable)


def conv_token_seqs(convs):
    """Each conv -> list of turns (prompt_tokens, full_tokens, resp_len). Token at position 0
    is the conv id (distinct -> no cross-conv prefix collision); later positions are a fixed
    pattern (values irrelevant to cost). Sequences grow per turn (prefix preserved)."""
    out = []
    for ci, conv in enumerate(convs):
        turns, cum = [], []
        for (p, r) in conv:
            need = p + r
            start = len(cum)
            ext = [(ci if (start + j) == 0 else _PAT + ((start + j) % 60000)) for j in range(need)]
            full = cum + ext
            prompt = full[: len(cum) + p]
            turns.append((prompt, full, r))
            cum = full
        out.append(turns)
    return out


def replay(model, cache, seqs, events):
    measure.reset_peak()
    ttfts, recomputes = [], 0
    dummy = mx.array([[7]], dtype=mx.int32)
    for e, (ci, ti) in enumerate(events):
        prompt, full, r = seqs[ci][ti]
        reused, remaining = cache.fetch_nearest_cache(_MKEY, prompt)
        if reused is None:
            pc = make_prompt_cache(model)
            remaining = prompt
            if ti > 0:
                recomputes += 1
        else:
            pc = reused
        t0 = time.perf_counter()
        if remaining:
            mx.eval(model(mx.array([remaining], dtype=mx.int32), cache=pc))  # prefill remainder
        ttfts.append((time.perf_counter() - t0) * 1e3)
        for _ in range(r):                                                   # decode response
            mx.eval(model(dummy, cache=pc))
        cache.insert_cache(_MKEY, full, pc)                                  # eviction fires here
        measure.free_buffers()
    a = np.asarray(ttfts)
    return dict(mean_ttft=float(a.mean()), p95_ttft=float(np.percentile(a, 95)),
                recomputes=recomputes, peak_mb=measure.peak_mb(), n_turns=len(events))


def run(args):
    from common import thermal
    _, line = thermal.power_state()
    print(f"[power] {line}")
    measure.set_mem_limit_gb(args.mem_limit_gb)
    cm = CostModel(args.model)
    print(f"[load] {models.resolve_model_id(args.model)} ...")
    model, _ = models.load_model(args.model)
    mx.eval(model.parameters())

    convs = workloads.load_sharegpt(args.trace, max_convs=args.scan)
    convs = [c for c in convs if len(c) >= 2 and sum(p + r for p, r in c) <= args.max_ctx][: args.n_convs]
    print(f"[trace] {len(convs)} convs (<= {args.max_ctx} tok, >= 2 turns)  {workloads.trace_stats(convs)}")
    seqs = conv_token_seqs(convs)
    events = build_schedule(convs, args.concurrency, args.think_mean)
    budget = int(args.kv_budget_gb * 1e9)
    print(f"[exp2a] {len(events)} turn-events | concurrency {args.concurrency} | "
          f"KV budget {args.kv_budget_gb} GB | K-sweep={args.ks}\n")

    def fmt(name, m):
        return (f"  {name:11s} | TTFT mean {m['mean_ttft']:8.1f}  p95 {m['p95_ttft']:9.1f} ms | "
                f"recomputes {m['recomputes']:4d} | peak {m['peak_mb']:6.0f} MB")

    Ks = [float(x) for x in str(args.ks).split(",")]
    res = {"lru": replay(model, LRUPromptCache(max_size=1 << 30, max_bytes=budget), seqs, events)}
    print(fmt("lru", res["lru"]))
    for K in Ks:
        time.sleep(args.cooldown)
        m = replay(model, CausalPromptCache(cm.prefill_ms, K=K, max_size=1 << 30,
                                            max_bytes=budget), seqs, events)
        res[f"causal_K{K:.0f}"] = m
        print(fmt(f"causal_K{K:.0f}", m))
    lru = res["lru"]["mean_ttft"]
    print(f"\n  vs LRU (TTFT mean {lru:.0f}ms, recomputes {res['lru']['recomputes']}):")
    for K in Ks:
        m = res[f"causal_K{K:.0f}"]
        print(f"    causal K={K:.0f}: {(1 - m['mean_ttft']/lru)*100:+5.1f}% TTFT, "
              f"{res['lru']['recomputes'] - m['recomputes']:+d} recomputes  "
              f"({'better' if m['mean_ttft'] < lru else 'WORSE'})")
    print("  (Exp3 sim: causal beat LRU. Real model needs K >> reuse-distance to protect active")
    print("   convs; small K thrashes (evicts active small convs -> recompute). Honest operating-point sweep.)")


def main():
    ap = argparse.ArgumentParser(description="Phase 2a: causal eviction on the real mlx-lm model.")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--trace", default=os.path.join(REPO, "data/raw/old/sg_52k.json"))
    ap.add_argument("--scan", type=int, default=4000, help="conversations to scan from the trace")
    ap.add_argument("--n-convs", type=int, default=40, help="filtered conversations to replay")
    ap.add_argument("--max-ctx", type=int, default=8000, help="filter: max total tokens / conv")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--think-mean", type=float, default=12.0)
    ap.add_argument("--kv-budget-gb", type=float, default=1.5)
    ap.add_argument("--ks", default="16,32,48", help="causal recency K sweep (K >> reuse-distance)")
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
