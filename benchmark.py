# BENCH PLAN (locked Jul 13): timing sibling of verify_decode.py — imports
# ModelRunner directly, no server, no HTTP. verify_decode asks "are the tokens
# right?"; this asks "how fast?". Same-workload-every-config is the whole point:
# every config must chew through the SAME 8 prompts at the SAME 32-token cap so
# tokens/sec is apples-to-apples across batch sizes.
#
# FIXED WORKLOAD: 8 hardcoded varied-length prompts, max_new_tokens=32 each,
# 256-token budget total. Temperature 0 everywhere so token counts are stable
# rep-to-rep (a rep that generates a different number of tokens isn't comparable).
#
# CONFIGS:
#   - generate_batch at batch sizes [1,2,4,8]. Batch N => split the 8 prompts
#     into 8/N sequential calls of N prompts each. Same total work every time.
#   - one context config: generate_text called 8× in a loop. Labeled as a
#     DIFFERENT decode path (sequential + KV-cache), NOT baseline-comparable to
#     the recompute batch path — it's here for reference, not head-to-head.
#
# PER CONFIG: 2 untimed warmup runs (first batch of that config), then 3 timed
# reps of the FULL workload. perf_counter() wraps ONLY the model calls. Per rep:
#   tokens/sec = tokens actually generated / wall seconds.
# Report mean/min/max tokens/sec + mean wall across the 3 reps.
#
# OUTPUT: aligned table to stdout + (a) timestamped JSON in results/raw/ with
# git hash / device / model / all raw per-rep numbers, and (b) an appended
# markdown section in results/RESULTS.md.
#
# DEVICE GUARD: print device at startup; this is T4-targeted, so refuse to run
# on CPU behind a warning-prompt unless --allow-cpu is passed.
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

from model_runner import ModelRunner

# ---------------------------------------------------------------------------- #
# Fixed workload — 8 varied-length prompts, one shared token cap.
# ---------------------------------------------------------------------------- #
PROMPTS = [
    "Hi",
    "The capital of France is",
    "Once upon a time",
    "In machine learning, the term overfitting refers to",
    "Write a short poem about the ocean at dawn.",
    "Explain how a transformer neural network processes a sequence of tokens, step by step.",
    "List three reasons why distributed systems are hard to debug in production environments.",
    "The quick brown fox jumps over the lazy dog and then",
]

MAX_NEW_TOKENS = 32          # per-prompt cap, every prompt, every config
TEMPERATURE = 0              # greedy => deterministic token counts across reps
BATCH_SIZES = [1, 2, 4, 8]   # generate_batch configs
WARMUP_RUNS = 2              # untimed, first-batch-only, per config
TIMED_REPS = 3               # timed repetitions of the full workload, per config

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
RESULTS_MD = os.path.join(RESULTS_DIR, "RESULTS.md")


# ---------------------------------------------------------------------------- #
# Workload runners — each returns tokens_generated for ONE full pass over the
# 8 prompts. perf_counter timing lives in the caller (time_config), so these do
# nothing but issue model calls and sum the token counts they report back.
# ---------------------------------------------------------------------------- #
def _chunks(prompts: list[str], n: int) -> list[list[str]]:
    """Split prompts into sequential groups of n (the last may be short if the
    total isn't divisible, but 8 is divisible by every batch size here)."""
    return [prompts[i:i + n] for i in range(0, len(prompts), n)]


def run_batch_workload(runner: ModelRunner, prompts: list[str], batch_size: int) -> int:
    """Full workload via generate_batch: 8/batch_size sequential calls, each of
    batch_size prompts, every prompt capped at MAX_NEW_TOKENS. Returns the total
    number of tokens actually generated across all prompts."""
    total = 0
    for chunk in _chunks(prompts, batch_size):
        caps = [MAX_NEW_TOKENS] * len(chunk)          # per-prompt list[int]
        for _text, n_tokens in runner.generate_batch(chunk, caps):
            total += n_tokens
    return total


def run_context_workload(runner: ModelRunner, prompts: list[str]) -> int:
    """Full workload via generate_text called once per prompt (8×) — the
    sequential + KV-cache decode path. Returns total tokens actually generated."""
    total = 0
    for prompt in prompts:
        _text, n_tokens = runner.generate_text(prompt, MAX_NEW_TOKENS, TEMPERATURE)
        total += n_tokens
    return total


# ---------------------------------------------------------------------------- #
# Timing harness — warmup, then TIMED_REPS timed passes; perf_counter wraps ONLY
# the workload call.
# ---------------------------------------------------------------------------- #
def time_config(label: str, warmup_fn, workload_fn) -> dict:
    """Warm up (untimed) then time TIMED_REPS full passes of workload_fn.

    warmup_fn():   runs the FIRST BATCH of this config, untimed (called WARMUP_RUNS×).
    workload_fn(): runs the FULL 8-prompt workload, returns tokens generated.

    Returns a config record: per-rep raw numbers + a mean/min/max summary.
    """
    print(f"\n[{label}]")
    for w in range(WARMUP_RUNS):
        print(f"  warmup {w + 1}/{WARMUP_RUNS} ...", flush=True)
        warmup_fn()

    reps = []
    for r in range(TIMED_REPS):
        start = time.perf_counter()
        tokens = workload_fn()
        wall = time.perf_counter() - start          # model calls ONLY
        tps = tokens / wall if wall > 0 else 0.0
        reps.append({
            "rep": r + 1,
            "wall_seconds": wall,
            "total_tokens": tokens,
            "tokens_per_sec": tps,
        })
        print(f"  rep {r + 1}/{TIMED_REPS}: {tokens} tok in {wall:.3f}s "
              f"= {tps:.2f} tok/s", flush=True)

    tps_vals = [rep["tokens_per_sec"] for rep in reps]
    wall_vals = [rep["wall_seconds"] for rep in reps]
    summary = {
        "mean_tokens_per_sec": sum(tps_vals) / len(tps_vals),
        "min_tokens_per_sec": min(tps_vals),
        "max_tokens_per_sec": max(tps_vals),
        "mean_wall_seconds": sum(wall_vals) / len(wall_vals),
    }
    return {"label": label, "reps": reps, "summary": summary}


def build_configs(runner: ModelRunner) -> list[dict]:
    """Assemble every config as (metadata, warmup_fn, workload_fn) and run it.

    Warmup uses the FIRST BATCH of each config: for batch size N that's the first
    N prompts through generate_batch; for the context config it's the first
    prompt through generate_text.
    """
    records = []

    # generate_batch at each batch size ------------------------------------- #
    for n in BATCH_SIZES:
        label = f"generate_batch (batch={n})"
        first_batch = _chunks(PROMPTS, n)[0]         # first N prompts

        def warmup_fn(fb=first_batch, bs=n):
            runner.generate_batch(fb, [MAX_NEW_TOKENS] * len(fb))

        def workload_fn(bs=n):
            return run_batch_workload(runner, PROMPTS, bs)

        rec = time_config(label, warmup_fn, workload_fn)
        rec.update({"kind": "batch", "batch_size": n, "num_calls": len(PROMPTS) // n})
        records.append(rec)

    # context config: generate_text 8× ------------------------------------- #
    ctx_label = "sequential+KV-cache — different decode path, not baseline-comparable"
    first_prompt = PROMPTS[0]

    def ctx_warmup_fn():
        runner.generate_text(first_prompt, MAX_NEW_TOKENS, TEMPERATURE)

    def ctx_workload_fn():
        return run_context_workload(runner, PROMPTS)

    ctx_rec = time_config(ctx_label, ctx_warmup_fn, ctx_workload_fn)
    ctx_rec.update({"kind": "context", "batch_size": 1, "num_calls": len(PROMPTS)})
    records.append(ctx_rec)

    return records


# ---------------------------------------------------------------------------- #
# Reporting — stdout table, JSON dump, RESULTS.md append.
# ---------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    """Current git commit hash, or None if unavailable (not a repo / no git)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _device_name(device: torch.device) -> str:
    """Human-readable device name (the actual GPU model on CUDA, else the type)."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device.index or 0)
    return device.type


def print_table(records: list[dict]) -> None:
    """Clean aligned table to stdout: config, mean/min/max tok/s, mean wall."""
    headers = ["Config", "mean tok/s", "min tok/s", "max tok/s", "mean wall (s)"]
    rows = []
    for rec in records:
        s = rec["summary"]
        rows.append([
            rec["label"],
            f"{s['mean_tokens_per_sec']:.2f}",
            f"{s['min_tokens_per_sec']:.2f}",
            f"{s['max_tokens_per_sec']:.2f}",
            f"{s['mean_wall_seconds']:.3f}",
        ])

    widths = [max(len(headers[c]), *(len(row[c]) for row in rows)) for c in range(len(headers))]
    # left-align the label column, right-align the numeric columns
    def fmt(cells):
        out = [cells[0].ljust(widths[0])]
        out += [cells[c].rjust(widths[c]) for c in range(1, len(cells))]
        return "  ".join(out)

    print("\n" + "=" * (sum(widths) + 2 * (len(widths) - 1)))
    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt(row))
    print("=" * (sum(widths) + 2 * (len(widths) - 1)))


def write_json(records: list[dict], meta: dict, stamp: str) -> str:
    """Dump the full run (meta + every raw per-rep number) to results/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"bench_{stamp}.json")
    payload = {
        **meta,
        "workload": {
            "n_prompts": len(PROMPTS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "token_budget": len(PROMPTS) * MAX_NEW_TOKENS,
            "warmup_runs": WARMUP_RUNS,
            "timed_reps": TIMED_REPS,
            "prompts": PROMPTS,
        },
        "configs": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def append_markdown(records: list[dict], meta: dict, json_path: str) -> None:
    """Append a summary section for this run to results/RESULTS.md."""
    # RESULTS.md is created by Task B, but stay robust if it's missing.
    if not os.path.exists(RESULTS_MD):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# RESULTS — development results log\n\n"
                    "## Benchmark runs\n\n")

    lines = []
    lines.append(f"\n### {meta['timestamp']} — Benchmark run · *auto-generated by benchmark.py*\n")
    lines.append(f"- **Device:** {meta['device']} (`{meta['device_type']}`)")
    lines.append(f"- **Git commit:** {meta['git_commit'] or 'unavailable'}")
    lines.append(f"- **Model:** `{meta['model_id']}` · torch {meta['torch_version']}")
    lines.append(f"- **Workload:** {len(PROMPTS)} prompts × {MAX_NEW_TOKENS} max_new_tokens "
                 f"({len(PROMPTS) * MAX_NEW_TOKENS}-token budget), temperature {TEMPERATURE}, "
                 f"{WARMUP_RUNS} warmup + {TIMED_REPS} timed reps")
    lines.append(f"- **Raw JSON:** `{os.path.relpath(json_path, RESULTS_DIR)}`\n")
    lines.append("| Config | mean tok/s | min tok/s | max tok/s | mean wall (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for rec in records:
        s = rec["summary"]
        lines.append(
            f"| {rec['label']} | {s['mean_tokens_per_sec']:.2f} | "
            f"{s['min_tokens_per_sec']:.2f} | {s['max_tokens_per_sec']:.2f} | "
            f"{s['mean_wall_seconds']:.3f} |"
        )
    lines.append("")

    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- #
# Device guard + entrypoint.
# ---------------------------------------------------------------------------- #
def device_guard(device: torch.device, allow_cpu: bool) -> None:
    """This benchmark is T4-targeted. On CPU, refuse behind a warning-prompt
    unless --allow-cpu was passed."""
    if device.type == "cuda":
        return
    print("\n" + "!" * 70)
    print(f"WARNING: device is '{device.type}', not a CUDA GPU.")
    print("This benchmark is T4-targeted; CPU numbers are slow and not comparable.")
    print("!" * 70)
    if allow_cpu:
        print("--allow-cpu set: continuing on CPU anyway.\n")
        return
    resp = input("Run the benchmark on CPU anyway? [y/N]: ").strip().lower()
    if resp != "y":
        print("Refusing to run on CPU. Pass --allow-cpu to override. Aborting.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Timing benchmark for ModelRunner.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Run even if the device is CPU (T4 is the intended target).")
    args = parser.parse_args()

    runner = ModelRunner()                              # constructed like verify_decode.py
    print(f"Device: {runner.device}  ({_device_name(runner.device)})")
    print(f"Model:  {runner.model_id}")
    device_guard(runner.device, args.allow_cpu)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    meta = {
        "timestamp": stamp,
        "git_commit": _git_commit(),
        "device": _device_name(runner.device),
        "device_type": runner.device.type,
        "torch_version": torch.__version__,
        "model_id": runner.model_id,
    }

    records = build_configs(runner)

    print_table(records)
    json_path = write_json(records, meta, stamp)
    append_markdown(records, meta, json_path)
    print(f"\nWrote raw JSON  -> {json_path}")
    print(f"Appended summary -> {RESULTS_MD}")


if __name__ == "__main__":
    main()
