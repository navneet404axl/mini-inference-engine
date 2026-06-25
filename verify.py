from model_runner import ModelRunner

runner = ModelRunner()
prompt = "The capital of France is"
max_tokens = 20

# --- YOUR loop (greedy: temperature=0) ---
your_text, n = runner.generate_text(prompt, max_tokens=max_tokens, temperature=0)
print("YOURS: ", your_text)

# --- HuggingFace's generate() (greedy: do_sample=False) ---
import torch
input_ids = runner.tokenizer(prompt, return_tensors="pt").input_ids.to(runner.device)
with torch.no_grad():
    out_ids = runner.model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,           # greedy
    )
# strip the prompt tokens off the front, decode only the new ones:
hf_new_ids = out_ids[0][input_ids.shape[1]:]
hf_text = runner.tokenizer.decode(hf_new_ids, skip_special_tokens=True)
print("HF:    ", hf_text)

print("MATCH? ", your_text.strip() == hf_text.strip())