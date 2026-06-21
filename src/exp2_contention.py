#!/usr/bin/env python3
"""
exp2_contention.py - Exp 2: CPU/GPU shared-bus bandwidth contention during decode.
[RQ2/H2] - the core novelty.  SKELETON - not implemented.  See EXPERIMENTS.md Exp 2.

Question: how much does concurrent CPU memory traffic slow GPU decode throughput on the
shared bus, and how far does it shift the keep-vs-recompute crossover N*?

Plan:
  (a) decode tok/s + effective GB/s for GPU decode ALONE
  (b) the same with a controlled CPU bandwidth load (STREAM-like, intensity sweep,
      pinned to P-cores) running concurrently
  -> slowdown curve + N* re-measured under contention.

Controls (RESEARCH.md §6): pin CPU load to P-cores; sweep intensity; thermal control is
especially important (sustained dual CPU+GPU load); powermetrics for effective GB/s.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import measure, thermal, models, workloads  # noqa: F401  (skeleton imports)


def decode_throughput(model, N, n_steps=64):
    """TODO: sustained decode tok/s over n_steps with N tokens cached (mx.eval per step).
    Reuse common.measure.time_decode internals; report tok/s + effective GB/s."""
    raise NotImplementedError


def start_cpu_bandwidth_load(intensity, pin_pcores=True):
    """TODO: launch a controlled CPU memory-bandwidth load (STREAM-like) at `intensity`,
    pinned to performance cores. Return a handle to stop it. (trap: P-core pin)"""
    raise NotImplementedError


def run(args):
    """TODO: for each N and CPU-load intensity, measure decode tok/s solo vs contended;
    log effective bandwidth (thermal.PowerMetricsLogger); emit contention curve + N* shift.
    Writes results/csv/exp2_<model>_<date>.csv and results/figures/exp2_*.png."""
    raise NotImplementedError("Exp2 not implemented - see EXPERIMENTS.md Exp 2")


def main():
    ap = argparse.ArgumentParser(description="Exp2 CPU/GPU bandwidth contention (SKELETON).")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--ns", default="512,2048,8192")
    ap.add_argument("--cpu-intensity", default="0,25,50,75,100", help="CPU load sweep (%)")
    ap.add_argument("--allow-battery", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
