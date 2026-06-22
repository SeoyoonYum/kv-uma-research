#!/usr/bin/env python3
"""
exp3_policygap.py - Exp 3: policy gap, current frameworks vs a measured-cost oracle.
[RQ3/H3]  Measurement-only SIMULATION (no model execution; scores decisions with the Exp1
cost model). See EXPERIMENTS.md Exp 3 (redefined 2026-06-21).

Replays multiturn traces (common.workloads) under four KV policies and measures the
headroom a UMA-aware policy reclaims. The keep-vs-recompute decision is made at each
inter-turn boundary (when a conversation goes idle): keep its KV resident (costs memory)
or drop it and recompute on the next turn (frees memory, costs a recompute). A per-
conversation KV budget = total_budget / concurrency models the memory pressure.

  rotating(W)       - mlx-lm default: cap cache at the last W tokens (cheap, bounded,
                      but SILENTLY DROPS older context -> coverage < 1).
  full_keep         - keep everything (cheap TTFT) but OOM when the conv's KV exceeds its
                      budget share (crash).
  always_recompute  - drop between turns (0 idle footprint, no OOM) but pay the full
                      recompute tax EVERY turn (high TTFT).
  oracle            - keep while it fits the budget (cheap, full context, no crash),
                      recompute ONLY the turns that would exceed budget -> best feasible.

Cost model: prefill_ms(N), decode_ms(N) interpolated from results/csv/exp1_<model>_*.csv;
KV bytes/token from the matching _meta.json. Pure arithmetic - no MLX, no model load.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import workloads  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "results", "csv")
FIG_DIR = os.path.join(REPO, "results", "figures")
POLICIES = ("rotating", "full_keep", "always_recompute", "oracle")


def _find_cost_files(model):
    csvs = [c for c in sorted(glob.glob(os.path.join(CSV_DIR, f"exp1_{model}_*.csv")))
            if "smoke" not in c and "_meta" not in c]
    metas = sorted(glob.glob(os.path.join(CSV_DIR, f"exp1_{model}_*_meta.json")))
    if not csvs or not metas:
        sys.exit(f"[exp3] need Exp1 cost data for {model} in results/csv/ (exp1_{model}_*.csv)")
    return csvs[-1], metas[-1]


class CostModel:
    """prefill_ms(N) / decode_ms(N) from the Exp1 curves (log-log interp + power-law
    extrapolation beyond the measured range), plus KV bytes/token."""

    def __init__(self, model):
        csv_path, meta_path = _find_cost_files(model)
        Ns, pre, dec = [], [], []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                Ns.append(int(r["N"])); pre.append(float(r["prefill_ms_med"]))
                dec.append(float(r["decode_ms_med"]))
        order = np.argsort(Ns)
        self.Ns = [Ns[i] for i in order]
        self.pre = [pre[i] for i in order]
        self.dec = [dec[i] for i in order]
        self.kv_bpt = json.load(open(meta_path))["kv_bytes_per_token"]
        self.src = os.path.basename(csv_path)

    def _loglog(self, N, ys):
        N = max(1.0, float(N))
        lx, lxs, lys = math.log(N), [math.log(x) for x in self.Ns], [math.log(y) for y in ys]
        if N <= self.Ns[0]:
            sl = (lys[1] - lys[0]) / (lxs[1] - lxs[0])
            return math.exp(lys[0] + sl * (lx - lxs[0]))
        if N >= self.Ns[-1]:
            sl = (lys[-1] - lys[-2]) / (lxs[-1] - lxs[-2])
            return math.exp(lys[-1] + sl * (lx - lxs[-1]))
        return math.exp(float(np.interp(lx, lxs, lys)))

    def prefill_ms(self, N):
        return self._loglog(N, self.pre) if N > 0 else 0.0

    def decode_ms(self, N):
        return self._loglog(N, self.dec)

    def incr_prefill_ms(self, a, b):
        """Cost to extend a cached prefix of `a` tokens to `b` (prefill the suffix)."""
        if b <= a:
            return 0.0
        return self.prefill_ms(b) - self.prefill_ms(a) if a > 0 else self.prefill_ms(b)

    def decode_sum_ms(self, ctx0, n):
        """Decode cost for n tokens over context growing from ctx0 (midpoint approx)."""
        return n * self.decode_ms(ctx0 + n / 2.0) if n > 0 else 0.0

    def kv_bytes(self, N):
        return self.kv_bpt * N


def simulate(conv, cm, policy, budget_bytes, W):
    """Replay one conversation under `policy`. Returns per-conversation metrics."""
    ctx = 0                       # tokens already in the conversation (before this turn)
    total, ttfts = 0.0, []
    peak_res, oom = 0, False
    covered, ctx_total = 0, 0
    for (p, r) in conv:
        Lq = ctx + p              # context length while answering this turn
        if policy == "full_keep":
            ttft = cm.incr_prefill_ms(ctx, Lq)
            resid = cm.kv_bytes(ctx + p + r)             # holds full context between turns
        elif policy == "always_recompute":
            ttft = cm.prefill_ms(Lq)                     # rebuild the whole context each turn
            resid = 0                                    # freed between turns
        elif policy == "rotating":
            ttft = cm.incr_prefill_ms(min(ctx, W), min(Lq, W))
            resid = cm.kv_bytes(min(ctx + p + r, W))     # capped window; older context dropped
        elif policy == "oracle":
            full = cm.kv_bytes(ctx + p + r)
            if full <= budget_bytes:
                ttft, resid = cm.incr_prefill_ms(ctx, Lq), full      # keep (cheap)
            else:
                ttft, resid = cm.prefill_ms(Lq), 0                   # recompute only when over budget
        else:
            raise ValueError(policy)

        total += ttft + cm.decode_sum_ms(Lq, r)
        ttfts.append(ttft)
        peak_res = max(peak_res, resid)
        if resid > budget_bytes:          # idle footprint exceeds the conv's budget share
            oom = True
        seen = min(Lq, W) if policy == "rotating" else Lq
        covered += seen; ctx_total += Lq
        ctx += p + r
    return dict(total_ms=total, mean_ttft_ms=float(np.mean(ttfts)),
                peak_res_mb=peak_res / 1e6, oom=oom,
                coverage=covered / ctx_total if ctx_total else 1.0)


def run(args):
    cm = CostModel(args.model)
    print(f"[exp3] cost model from {cm.src} (KV {cm.kv_bpt} B/tok)")
    if args.trace and os.path.exists(args.trace):
        convs = workloads.load_sharegpt(args.trace, max_convs=args.max_convs)
        src = os.path.basename(args.trace)
    else:
        convs = workloads.synthetic_conversations(n_convs=args.n_convs, seed=0)
        src = f"synthetic(n={args.n_convs})"
        if args.trace:
            print(f"[exp3] trace '{args.trace}' not found -> using synthetic")
    print(f"[exp3] traces: {src}  {workloads.trace_stats(convs)}")
    budget_bytes = args.kv_budget_gb * 1e9 / args.concurrency
    print(f"[exp3] per-conversation KV budget = {budget_bytes/1e6:.0f} MB "
          f"({args.kv_budget_gb} GB / concurrency {args.concurrency}); rotating W={args.window}\n")

    agg, rows = {}, []
    for pol in [p for p in POLICIES if p in args.policies.split(",")]:
        ms = [simulate(c, cm, pol, budget_bytes, args.window) for c in convs]
        agg[pol] = dict(
            mean_total_ms=float(np.mean([m["total_ms"] for m in ms])),
            mean_ttft_ms=float(np.mean([m["mean_ttft_ms"] for m in ms])),
            p95_ttft_ms=float(np.percentile([m["mean_ttft_ms"] for m in ms], 95)),
            mean_peak_res_mb=float(np.mean([m["peak_res_mb"] for m in ms])),
            oom_rate=float(np.mean([m["oom"] for m in ms])),
            mean_coverage=float(np.mean([m["coverage"] for m in ms])))
        rows.append(dict(policy=pol, **{k: round(v, 3) for k, v in agg[pol].items()}))
        a = agg[pol]
        print(f"  {pol:17s} | total {a['mean_total_ms']:8.0f}ms | TTFT mean {a['mean_ttft_ms']:7.1f} "
              f"p95 {a['p95_ttft_ms']:7.1f} ms | footprint {a['mean_peak_res_mb']:6.0f}MB | "
              f"OOM {a['oom_rate']*100:4.0f}% | ctx {a['mean_coverage']*100:4.0f}%")

    analyze(agg)
    date = datetime.now().strftime("%Y%m%d")
    os.makedirs(CSV_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(CSV_DIR, f"exp3_{args.model}_{date}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n[out] wrote {path}")
    plot(agg, os.path.join(FIG_DIR, f"exp3_policygap_{args.model}_{date}.png"), args.model)


def analyze(agg):
    print("\n================ POLICY GAP vs ORACLE ================")
    if "oracle" not in agg:
        print("(no oracle in policy set)"); return
    o = agg["oracle"]
    for pol, a in agg.items():
        if pol == "oracle":
            continue
        dl = a["mean_total_ms"] / o["mean_total_ms"]
        note = []
        if a["oom_rate"] > o["oom_rate"] + 0.01:
            note.append(f"OOM {a['oom_rate']*100:.0f}% (oracle {o['oom_rate']*100:.0f}%)")
        if a["mean_coverage"] < o["mean_coverage"] - 0.01:
            note.append(f"drops {(1-a['mean_coverage'])*100:.0f}% of context")
        if dl > 1.05:
            note.append(f"{dl:.2f}x latency")
        print(f"  {pol:17s}: {'; '.join(note) if note else 'matches oracle'}")
    print("=> oracle = keep when it fits budget, recompute only when over -> full context,")
    print("   no crash, recompute tax paid ONLY under pressure. rotating hides context loss;")
    print("   full_keep crashes; always_recompute over-pays. This headroom motivates Phase 2.")
    print("=====================================================")


def plot(agg, png, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pols = list(agg.keys())
    colors = {"rotating": "tab:gray", "full_keep": "tab:blue",
              "always_recompute": "tab:orange", "oracle": "tab:green"}
    cs = [colors.get(p, "tab:gray") for p in pols]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("mean_total_ms", "mean total latency / conv (ms)"),
              ("mean_ttft_ms", "mean TTFT (ms)"),
              ("oom_rate", "OOM / crash rate"),
              ("mean_coverage", "context coverage")]
    for axi, (key, title) in zip(ax.ravel(), panels):
        vals = [agg[p][key] * (100 if key in ("oom_rate", "mean_coverage") else 1) for p in pols]
        axi.bar(range(len(pols)), vals, color=cs)
        axi.set_xticks(range(len(pols))); axi.set_xticklabels(pols, rotation=20, fontsize=9)
        axi.set_title(title); axi.grid(True, axis="y", alpha=0.25)
        if key in ("oom_rate", "mean_coverage"):
            axi.set_ylabel("%")
    fig.suptitle(f"Exp3 policy gap (cost-model sim) - {model}", fontsize=13)
    fig.tight_layout(); fig.savefig(png, dpi=150)
    print(f"[out] wrote {png}")


def main():
    ap = argparse.ArgumentParser(description="Exp3 policy gap (measurement-only simulation).")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--trace", default="", help="ShareGPT/LMSys JSON path; empty -> synthetic")
    ap.add_argument("--max-convs", type=int, default=None)
    ap.add_argument("--n-convs", type=int, default=400, help="synthetic conversation count")
    ap.add_argument("--kv-budget-gb", type=float, default=8.0, help="total KV budget")
    ap.add_argument("--concurrency", type=int, default=16, help="budget share = budget/concurrency")
    ap.add_argument("--window", type=int, default=4096, help="rotating cache window W (tokens)")
    ap.add_argument("--policies", default=",".join(POLICIES))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
