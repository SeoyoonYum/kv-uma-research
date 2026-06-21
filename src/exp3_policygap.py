#!/usr/bin/env python3
"""
exp3_policygap.py - Exp 3: policy gap, current frameworks vs a measured-cost oracle.
[RQ3/H3]  SKELETON - not implemented.  See EXPERIMENTS.md Exp 3.

Replay multi-turn chat traces under four policies and measure the headroom a UMA-aware
policy could reclaim:
  (a) mlx-lm default rotating   (b) full-keep   (c) always-recompute
  (d) oracle = pick keep/recompute by the Exp1(/Exp2) measured cost model each decision.

Metrics: TTFT, inter-token latency, peak memory, throughput, OOM/crash rate.
Needs: common.workloads.load_multiturn_trace (ShareGPT / LMSys-Chat).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import measure, models, workloads  # noqa: F401  (skeleton imports)

POLICIES = ("rotating", "full_keep", "always_recompute", "oracle")


def oracle_decision(N, cost_model):
    """TODO: cheaper of keep-and-read vs drop-and-recompute at context N, from the Exp1
    measured cost model (+ Exp2 contention term). Returns 'keep' or 'recompute'."""
    raise NotImplementedError


def replay_trace(model, trace, policy):
    """TODO: replay one multi-turn conversation under `policy`; record TTFT, inter-token
    latency, peak memory, throughput, and whether it OOM'd / crashed."""
    raise NotImplementedError


def run(args):
    """TODO: load traces (common.workloads.load_multiturn_trace), replay under each policy,
    aggregate; emit per-policy table + default-vs-oracle gap plot.
    Writes results/csv/exp3_<model>_<date>.csv and results/figures/exp3_*.png."""
    raise NotImplementedError("Exp3 not implemented - see EXPERIMENTS.md Exp 3")


def main():
    ap = argparse.ArgumentParser(description="Exp3 policy gap (SKELETON).")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--trace", default="data/traces/")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--allow-battery", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
