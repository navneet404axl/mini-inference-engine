# VERIFY PLAN (locked Jul 8): compares full generated token-ID sequences per prompt,
# batched generate_batch vs sequential generate_tokens, temperature 0.
# PREDICTION: identical sequences. If ONE prompt forks: find first divergent token,
# check top-2 logit gap (tiny gap = fp16 noise). ALL prompts forking = my bug
# (mask growth / positions / feedback).
# STEP 1: same 4 different-length prompts as verify_prefill
# STEP 2: sequential baseline — full token-ID list per prompt via generate_tokens (temp 0, N tokens)
# STEP 3: batched — generate_batch, same prompts, same N
#         NOTE: generate_batch returns decoded TEXT; decide comparison strategy —
#         either add an IDs-return debug path or compare decoded text of sequential IDs
# STEP 4: per-prompt PASS/FAIL + first-divergence index on FAIL
from model_runner import ModelRunner

# Same 4 different-length prompts as verify_prefill.py (copied exactly).
PROMPTS = [
    "Hi",
    "The capital of France is",
    "In machine learning, the term overfitting refers to",
    "Once",
]

N_TOKENS = 20

def main():
    runner = ModelRunner()  # constructed exactly as verify_prefill.py does

    # SEQUENTIAL ROAD — one prompt at a time, no padding.
    sequential_ids = []
    for p in PROMPTS:
        ids = runner.tokenizer(p, return_tensors="pt").input_ids.to(runner.device)
        seq_ids = runner.generate_tokens(ids, max_tokens=N_TOKENS, temperature=0)  # FULL list
        # EOS-convention strip (sequential collects trailing EOS; batched bins never do).
        if seq_ids and seq_ids[-1] == runner.eos_token_id:
            seq_ids = seq_ids[:-1]
        sequential_ids.append(seq_ids)

    # BATCHED ROAD — one call, raw ID bins.
    # Stage 4: max_new_tokens is now per-prompt list[int]; replicate the single cap
    # across all prompts to preserve the original 20-token verification semantics.
    batched_ids = runner._generate_batch_ids(PROMPTS, max_new_tokens=[N_TOKENS] * len(PROMPTS))

    # COMPARE per prompt on the full ID lists.
    all_pass = True
    for p, s, b in zip(PROMPTS, sequential_ids, batched_ids):
        ok = (s == b)
        all_pass &= ok
        print(f"{'PASS' if ok else 'FAIL'} | seq_len={len(s)} vs batch_len={len(b)} | {p!r}")
        if not ok:
            div = next((i for i, (si, bi) in enumerate(zip(s, b)) if si != bi), None)
            if div is None:
                print(f"  length-only mismatch: {len(s)} vs {len(b)} (common prefix identical)")
            else:
                print(f"  first divergence index: {div}")
            print(f"  seq={s}")
            print(f"  batch={b}")

    print("\nALL PASS ✅" if all_pass
          else "\nMISMATCH ❌ — suspect mask growth / positions / feedback")

if __name__ == "__main__":
    main()
