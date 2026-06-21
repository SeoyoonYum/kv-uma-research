# data/

- `raw/` — raw inputs (datasets, downloads). **Gitignored** (not versioned); recreate locally.
- `traces/` — multi-turn chat traces for Exp3 policy replay (ShareGPT / LMSys-Chat).
  Loaded by `src/common/workloads.load_multiturn_trace`. Drop trace files here.
