# results/

Where measurements live.

- **`raw/`** — machine-written JSON, one file per benchmark run
  (`bench_YYYY-MM-DD_HHMM.json`). These are the source-of-truth raw numbers:
  git commit, device, model id, full config details, and every per-rep
  measurement. Never hand-edit these.
- **`RESULTS.md`** — the human log. Append-only, newest at the bottom.
  `benchmark.py` appends a summary section here automatically after each run,
  and **manual entries are welcome** for non-benchmark milestones (verification
  runs, seam checks, one-off floods) that `benchmark.py` doesn't produce.

So: `benchmark.py` writes both a JSON file into `raw/` and a markdown summary
into `RESULTS.md`; you write into `RESULTS.md` by hand for everything else.
