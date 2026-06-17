# Working Agreement for Claude Code

## Core rule
This is a learning project. I (the human) must understand and be able to
rebuild every core piece. Move fast on plumbing, slow on the parts that teach.

## Do NOT write (I write these myself — review/critique only):
- The decode / token generation loop
- The request queue logic and backpressure handling
- The batching logic
- Metrics and percentile (p50/p95/p99) math
- The benchmark harness design

## Always do this for the parts I write myself:
Instead of writing the implementation, scaffold it as STUBS for me to fill in:
- Write the function/class name and signature
- Write a docstring explaining what it should do, the inputs, and the outputs
- Add inline comments marking each step: "# STEP 1: ... (your logic here)"
- Explain WHERE my logic goes and WHAT approach to consider — but do NOT
  write the actual logic. Leave it for me.
- If I ask, give hints or pseudocode, but let me write the real code.

Example of what I want:
    def generate_tokens(model, input_ids, max_tokens, temperature):
        """
        Run the autoregressive decode loop.
        Input: model, starting token IDs, how many to generate, temperature.
        Output: list of generated token IDs.
        """
        # STEP 1: set up the KV cache / initial state (your logic here)
        # STEP 2: loop max_tokens times (your logic here)
        #   - run a forward pass on the latest token
        #   - take logits[-1], apply temperature, softmax, sample
        #   - append the new token, feed it back
        # STEP 3: return the generated tokens
        pass

## Free to fully write (boilerplate — just do it):
- FastAPI route scaffolding and request/response models (Pydantic)
- Dockerfile, config files, requirements.txt
- Logging setup, project structure
- Plotting/charting code for benchmarks

## Before implementing anything non-trivial:
Ask me for my design first. Tell me the tradeoffs, let me decide.
