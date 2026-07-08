# verify_prefill.py — batched prefill vs sequential, temp 0, exact-match check
from model_runner import ModelRunner

# STEP 1: prompts of DIFFERENT lengths — this is load-bearing.
# Same lengths = no padding = you verified nothing.
PROMPTS = [
    "Hi",
    "The capital of France is",
    "In machine learning, the term overfitting refers to",
    "Once",
]

def main():
    runner = ModelRunner()  # adapt if your constructor takes args (model name, device, etc.)

    # STEP 2: sequential baseline — first generated token ID per prompt
    sequential_ids = []
    for p in PROMPTS:
        ids = runner.tokenizer(p, return_tensors="pt").input_ids.to(runner.device)
        first_id = runner.generate_tokens(ids, max_tokens=1, temperature=0)[0]
        sequential_ids.append(first_id)

    # STEP 3: batched — all prompts in one call
    batched_ids = runner.generate_batch(PROMPTS, max_new_tokens=1)

    # STEP 4: compare per-prompt
    all_pass = True
    for p, s, b in zip(PROMPTS, sequential_ids, batched_ids):
        ok = (s == b)
        all_pass &= ok
        print(f"{'PASS' if ok else 'FAIL'} | seq={s} batch={b} | {p!r}")

    print("\nALL PASS ✅" if all_pass else "\nMISMATCH ❌ — suspect padding/mask/positions")

if __name__ == "__main__":
    main()