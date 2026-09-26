# VERIFY PLAN: continuous batching must be numerically invisible.
# Each request's greedy token IDs from ContinuousBatch must equal the
# single-sequence generate_tokens path (temperature 0), even though requests
# JOIN and LEAVE the running batch at different steps.
#
# The schedule below is built to hit every membership-change path:
#   - a join into an EMPTY batch (A, B at step 0)
#   - joins into a RUNNING batch with a LONGER cache (C at step 3) and with
#     a SHORTER cache (D, a long prompt, at step 6), so both padding sides
#     of _merge are exercised
#   - different budgets, so rows LEAVE at different steps (evict), including
#     the longest-cache row leaving first, which forces a TRIM
#   - a max_new_tokens=1 request (E), which finishes inside add() and never
#     enters the batch
#   - a request added AFTER others left (F), so it merges into a trimmed cache
#
# PREDICTION: all identical. If only rows that JOINED A RUNNING BATCH fail,
# suspect position_ids or the left-pad merge. If rows fail only AFTER someone
# LEFT, suspect _select / _trim.
from continuous_batch import ContinuousBatch
from model_runner import ModelRunner

# (name, prompt, max_new_tokens, step at which it is admitted)
SCHEDULE = [
    ("A", "Hi", 12, 0),
    ("B", "The capital of France is", 20, 0),
    ("C", "Once", 8, 3),
    # D has the LONGEST prompt but a small budget: it leaves while A/B are
    # still running, so its extra columns become all-pad and must be TRIMMED
    # (watch cache_len drop in the step log).
    ("D", "In machine learning, the term overfitting refers to the situation where a model", 4, 6),
    ("E", "Water boils at", 1, 6),
    ("F", "The quick brown fox", 10, 22),
]


def sequential(runner: ModelRunner, prompt: str, n: int) -> list[int]:
    ids = runner.tokenizer(prompt, return_tensors="pt").input_ids.to(runner.device)
    out = runner.generate_tokens(ids, max_tokens=n, temperature=0)
    # Sequential collects a trailing EOS; the batch paths never do.
    if out and out[-1] == runner.eos_token_id:
        out = out[:-1]
    return out


def main() -> None:
    runner = ModelRunner()
    expected = {name: sequential(runner, p, n) for name, p, n, _ in SCHEDULE}

    engine = ContinuousBatch(runner)
    got: dict[str, list[int]] = {}
    streamed: dict[str, list[int]] = {name: [] for name, *_ in SCHEDULE}

    def on_token(key, token_id):
        streamed[key].append(token_id)

    step = 0
    pending = sorted(SCHEDULE, key=lambda r: r[3])
    while pending or len(engine):
        arrivals = [r for r in pending if r[3] <= step]
        pending = [r for r in pending if r[3] > step]
        if arrivals:
            for key, ids in engine.add([(n, p, m, 0.0) for n, p, m, _ in arrivals], on_token):
                got[key] = ids
        for key, ids in engine.step(on_token):
            got[key] = ids
        cache_len = engine.mask.shape[1] if len(engine) else 0
        print(f"step {step:2d}: running={engine.keys} cache_len={cache_len}")
        step += 1

    all_pass = True
    print()
    for name, prompt, n, at in SCHEDULE:
        ok = got.get(name) == expected[name] and streamed[name] == expected[name]
        all_pass &= ok
        print(f"{'PASS' if ok else 'FAIL'} | {name} joined@{at:2d} max={n:2d} "
              f"len={len(got.get(name, []))} | {prompt[:40]!r}")
        if not ok:
            print(f"  expected={expected[name]}")
            print(f"  got     ={got.get(name)}")
            print(f"  streamed={streamed[name]}")

    print("\nALL PASS ✅" if all_pass else "\nMISMATCH ❌ — see plan notes at top of file")


if __name__ == "__main__":
    main()
