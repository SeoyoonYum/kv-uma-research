# kv-uma-research

Single-pool unified-memory (Apple Silicon) KV-cache *decision* policy — keep / recompute / evict — and the coordinate where PCIe-era cost models break. Phase 1 measures that the cost structure differs (Exp1–3); Phase 2 builds a policy that beats current frameworks. Every experiment must run on the target machine (M4 MacBook Air, 16 GB unified memory) because the hardware is the measurement subject.

## Navigation
- **[RESEARCH.md](RESEARCH.md)** — master context / source of truth. Read this first every session.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — the full experiment program (Exp1–4 + Phase 2 policy).
- **[DECISIONS.md](DECISIONS.md)** — dated decision log (research journal).
- **[RELATED_WORK.md](RELATED_WORK.md)** — tiered reading list with notes.
- `src/` — measurement code. `src/common/` = shared infra (timing, thermal, model registry, workloads).
- `results/{csv,figures,logs}/` — outputs. `data/{raw,traces}/` — inputs (raw is gitignored).
- `policy/` — Phase 2 adaptive policy (after the Go/No-Go gate). `paper/` — outline.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/exp1_costcurve.py --model 1.5B          # -> results/csv + results/figures
python src/exp1_costcurve.py --model 1.5B --smoke  # fast sanity (N=128,256)
```

Verified environment: mlx 0.31.2 / mlx-lm 0.31.3 / numpy 2.4.6 / matplotlib 3.11.0 / Python 3.13 (see RESEARCH.md §7 STATUS). mlx-lm APIs are version-sensitive (RESEARCH.md §6) — `src/common/measure.py` resolves them defensively.
