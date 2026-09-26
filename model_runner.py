"""
model_runner.py

=============================================================================
WHAT THIS FILE IS
=============================================================================
Everything that touches the weights. Loads Qwen2.5-1.5B onto the GPU once and
exposes three ways to decode from it:

    generate_text / generate_tokens   single prompt, blocking, returns all ids
    stream_tokens                     single prompt, generator, yields text
    generate_batch / _generate_batch_ids
                                      N prompts, ONE forward pass per step

It knows nothing about HTTP, queues, batching policy, or metrics — main.py
decides WHEN and WITH WHOM to run; this file only knows HOW to run.

=============================================================================
HOW WE GOT HEREffƒ
=============================================================================
Step 1 — generate_tokens: the autoregressive loop with a KV cache. The single
    idea the whole project rests on: after the first pass over the prompt, you
    never re-run the prompt again — you feed back one token and the cache.
Step 2 — stream_tokens: the same loop restructured as a generator so tokens
    can leave the process as they are produced instead of after the last onfe.
Step 3 (Jul 5-9) — _generate_batch_ids: N sequences decoded together. This is
    where the throughput came from, and where all the fiddly parts live:
    left padding, an attention mask that grows each step, per-sequence token
    budgets, and per-sequence EOS handling.
    It was first written as a RECOMPUTE loop (re-feed the whole grown
    input_ids every step) because that version is obviously correct and easy
    to reason about. Once its output was verified, the KV cache was ported in
    — feed only the new token column, keep past_key_values. Same outputs,
    massively less work per step.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Override via env vars without touching code.
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B")


def _pick_device(requested: str | None = None) -> torch.device:
    """Resolve which device to load onto. Prefer CUDA (the T4 target); fall
    back to CPU so the code still runs for local dev when no GPU is attached.
    An explicit `requested` always wins, for tests that need to pin a device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA not available — falling back to CPU. Generation will be slow.")
    return torch.device("cpu")


def _pick_dtype(device: torch.device) -> torch.dtype:
    """fp16 on GPU (T4 supports it; halves memory vs fp32 and roughly doubles
    throughput), fp32 on CPU — CPU fp16 is emulated and slower, not faster."""
    return torch.float16 if device.type == "cuda" else torch.float32


class ModelRunner:
    """Owns the tokenizer + model and runs generation.

    Construct once at server startup and reuse for every request — loading the
    weights is the expensive part (seconds, plus GBs of VRAM) and must not
    happen per-request. main.py enforces this by creating exactly one instance
    in the lifespan hook and stashing it on app.state.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = _pick_device(device)
        self.dtype = dtype or _pick_dtype(self.device)

        logger.info("Loading tokenizer for %s ...", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(
            "Loading model %s onto %s (%s) ...", model_id, self.device, self.dtype
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()  # inference only — disable dropout etc.

        # Qwen ships an explicit eos token but no pad token. Batching REQUIRES
        # a pad id (short prompts have to be filled out to the longest one), so
        # reuse eos: padding positions are masked out by the attention mask
        # anyway, which makes the choice of filler value irrelevant to the math.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Model ready.")

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # Single-prompt path: text -> ids -> decode loop -> ids -> text.      #
    # ------------------------------------------------------------------ #
    def generate_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int]:
        """Tokenize a prompt, run the decode loop, and detokenize the result.

        This thin wrapper exists so the decode loop can stay purely in token-id
        space — it never sees strings, which keeps it testable without the
        tokenizer.

        Returns (generated_text, num_tokens_generated). The generated text is
        ONLY the new tokens — the prompt is not echoed back, which is why the
        loop below returns new ids rather than the full sequence.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        new_token_ids = self.generate_tokens(
            input_ids=input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return text, len(new_token_ids)

    # ------------------------------------------------------------------ #
    # The core autoregressive loop (Step 1 of the project).               #
    # ------------------------------------------------------------------ #
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
    ) -> list[int]:
        """
        Run the autoregressive decode loop with a KV cache.

        Input:
          - input_ids:   LongTensor of shape (1, prompt_len) already on self.device
          - max_tokens:  max number of NEW tokens to generate
          - temperature: sampling temperature (0 -> greedy / deterministic)
        Output:
          - list[int] of the NEW token ids only (the prompt is not included)

        SHAPE OF THE LOOP
          One PREFILL pass over the whole prompt builds the cache and gives the
          logits for the last position. Then each DECODE step: read the last
          position's logits, turn them into a token, and feed back that ONE
          token plus the cache.

        WHY THE KV CACHE IS THE WHOLE POINT
          Without it, step n has to re-run attention over all n prior tokens,
          so generating k tokens costs O(k^2) work. past_key_values stores the
          per-layer keys/values already computed, so each step only computes
          the new token's row: O(k). This is the difference between a toy and
          something that can serve.

        SAMPLING
          temperature == 0 is a separate branch, not "temperature very small":
          dividing logits by 0 is inf/NaN. Greedy = argmax = fully
          deterministic. Otherwise logits/T reshapes the distribution (T < 1
          sharpens toward the top token, T > 1 flattens it), softmax turns it
          into probabilities, multinomial draws one.

        EOS NOTE
          The eos id is appended to `generated` BEFORE breaking, so it counts
          toward tokens_generated. It never appears in the returned text
          because generate_text decodes with skip_special_tokens=True. This
          differs from stream_tokens, which stops before emitting it — see
          there for why the streaming case has to.

        torch.no_grad(): inference only. Without it every forward pass would
        build an autograd graph we never use — pure memory and time.
        """
        with torch.no_grad():
            # PREFILL — one pass over the full prompt to populate the cache.
            outputs = self.model(input_ids, use_cache=True)
            past_key_values =  outputs.past_key_values
            generated = []                          # collect new token ids here

            for i in range(max_tokens):
                # logits is (batch, seq_len, vocab); the prediction for the NEXT
                # token always lives at the LAST position.
                next_token_logits = outputs.logits[:,-1,:]

                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    scaled = next_token_logits / temperature
                    probs = torch.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()         # tensor -> plain int
                generated.append(token_id)

                if token_id == self.eos_token_id:    # stop early
                    break
                # DECODE STEP — feed back ONLY the new token; the cache supplies
                # everything before it. Refresh the cache from the new outputs,
                # or the next step re-attends against a stale prefix.
                outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
        return generated

    # ------------------------------------------------------------------ #
    # Streaming twin of generate_tokens (Step 2): identical loop, but      #
    # YIELDS decoded text per token instead of collecting ids into a list. #
    # ------------------------------------------------------------------ #
    def stream_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
    ):
        """
        Autoregressive decode loop as a GENERATOR — yields text incrementally.

        Why a generator at all: it inverts control. The caller (the SSE
        endpoint in main.py) drives the loop one step at a time and can write
        each token to the socket the moment it exists, so the user sees output
        after ~1 token of latency instead of after all N.

        Input:
          - input_ids:   LongTensor of shape (1, prompt_len) already on self.device
          - max_tokens:  max number of NEW tokens to generate
          - temperature: sampling temperature (0 -> greedy)
        Yields:
          - str: the decoded text for each new token, one at a time

        THE ONE REAL DIFFERENCE FROM generate_tokens: the EOS check happens
        BEFORE the yield, so the caller never receives the end-of-sequence
        marker. Streaming has no post-processing stage — there is no final
        decode(skip_special_tokens=True) to strip it, because each token is
        decoded and shipped in isolation. So it has to be filtered here.

        Also note this path is per-token decode(), not one decode() of the
        whole sequence, which is why multi-token characters can look slightly
        different from the blocking path's output.
        """
        with torch.no_grad():
            # PREFILL — same as generate_tokens.
            outputs = self.model(input_ids, use_cache=True)
            past_key_values =  outputs.past_key_values
            for i in range(max_tokens):
                next_token_logits = outputs.logits[:,-1,:]

                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    scaled = next_token_logits / temperature
                    probs = torch.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()         # tensor -> plain int


                if token_id == self.eos_token_id:    # stop early, BEFORE yielding
                    break
                yield self.tokenizer.decode([token_id])
                # Same cache handoff as generate_tokens: one token in, refreshed
                # past_key_values out.
                outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

    # ------------------------------------------------------------------ #
    # Batched decode (Step 3) — the throughput win.                       #
    # ------------------------------------------------------------------ #
    def _generate_batch_ids(
        self,
        prompts: list[str],
        max_new_tokens: list[int],
        on_token: Callable[[int, int], None] | None = None,
    ) -> list[list[int]]:
        """Decode N prompts together, one forward pass per step for the whole batch.

        Input:  prompts (len N) and max_new_tokens (len N, per-request budgets).
                on_token (optional) — per-token hook for streaming (Stage 4,
                step 1.2). Called as on_token(i, token_id) the moment row i
                COLLECTS a token, i.e. only for tokens that will also land in
                generated_tokens[i] (never EOS, never a finished row). None =
                no streaming; the loop must behave exactly as before.
                It runs on WHATEVER THREAD this method runs on (the worker's
                executor thread), so it must never touch asyncio objects
                directly — that is the caller's problem, not this file's.
        Output: N lists of new token ids, IN THE SAME ORDER as `prompts`.

        ORDER IS A CONTRACT. main.py's worker zips these results back onto the
        batch positionally — nothing here may ever sort or reorder sequences.

        WHY THIS IS FASTER: a forward pass at batch size 8 costs almost the
        same wall-clock as batch size 1 (the GPU is memory-bandwidth bound on
        the weights, which get read once either way). So batching is close to
        free throughput — that is the entire reason the queue and worker in
        main.py exist.

        THE FOUR THINGS BATCHING FORCES YOU TO HANDLE
          1. Prompts have different lengths -> pad, and pad on the LEFT.
          2. Padding must not be attended to -> attention_mask, which has to
             grow by one column every step.
          3. Sequences hit EOS at different times -> a `finished` flag vector
             instead of a single break.
          4. Requests asked for different token counts -> per-sequence budgets
             checked each step, with the loop bounded by the largest.

        Returns raw id bins, not text (changed Jul 9): splitting decode out of
        the loop made it possible to verify the ids directly against the
        single-prompt path, and lets the public wrapper own detokenization.
        """
        # STEP 1: tokenizer config for batching.
        # padding_side = "left" is the critical one. Decoding always reads
        # logits at position -1; with RIGHT padding, position -1 of a short
        # sequence is a pad token and you would sample from garbage. With LEFT
        # padding, every real sequence ends at the same final position, so
        # `logits[:, -1, :]` is the correct next-token distribution for ALL
        # rows at once.
        # TODO Jul 5: move tokenizer config to __init__ — mutating shared state per-call is a smell
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # STEP 2: one padded batch. padding=True pads to the longest prompt in
        # THIS batch (not a fixed max), so a batch of short prompts stays cheap.
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # STEP 3: PREFILL for the whole batch — one pass over all prompts.
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache = True,
            )
            past_key_values = outputs.past_key_values
            # STEP 4: next-token logits for every sequence. Left padding is what
            # makes this single slice valid for all rows (see STEP 1).
            scores = outputs.logits[:, -1, :]
            # STEP 5: greedy pick. KNOWN LIMITATION: the batched path ignores
            # per-request temperature — API accepts it, this path is argmax.
            # Deliberate: correctness of batching was verified before adding
            # sampling. Shape stays flat [batch] — the "home shape" for tokens
            # throughout this loop; it is unsqueezed to [batch, 1] exactly once,
            # at the point the model needs a sequence dimension.
            next_token_ids = torch.argmax(scores, dim=-1)  # shape: [batch]

            # STEP 6: bookkeeping.
            #   generated_tokens — one output bin per sequence, index-aligned to prompts
            #   finished         — per-sequence latch: once True it stays True
            batch_size = inputs["input_ids"].shape[0]
            generated_tokens = [[] for _ in range(batch_size)]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            attention_mask = inputs["attention_mask"]

            # STEP 7: the batched decode loop.
            #
            # DESIGN (locked Jul 7) — "keep and ignore": a sequence that finishes
            # is NOT removed from the batch. Its row keeps being computed and the
            # result is simply not collected. Evicting rows mid-flight would mean
            # re-indexing the cache, the mask and the bins every time one ends —
            # far more complexity than the wasted compute is worth at this size.
            # (Evicting/refilling instead is what "continuous batching" does.)
            #
            # The loop runs to the LARGEST request's budget; shorter requests are
            # latched finished before that.
            max_steps = max(max_new_tokens)
            for step in range(max_steps):

                # MOVE 1 — BUDGET: latch any sequence that has already produced
                # everything its request asked for.
                # Check the cap before COLLECT so a request with max_new_tokens == 0 emits no token.
                # This may cause one harmless extra loop iteration, which is an accepted tradeoff.
                for i in range(batch_size):
                    if len(generated_tokens[i]) >= max_new_tokens[i]:
                        finished[i] = True
                is_eos = next_token_ids == self.eos_token_id    # bool [batch]
                # MOVE 2 — COLLECT: only for rows that are neither finished nor
                # emitting EOS. This is where "keep and ignore" is enforced —
                # finished rows still compute, they just stop contributing.
                for i in range(batch_size):
                    if not finished[i] and not is_eos[i]:
                        token_id = next_token_ids[i].item()
                        generated_tokens[i].append(token_id)
                        # STREAM HOOK (Stage 4): the one spot that already knows
                        # row -> token AND has filtered EOS/finished rows, so the
                        # hook sees exactly the tokens that land in the bins.
                        # Reuses the int above — a second .item() would be a
                        # second GPU->CPU sync per token.
                        if on_token is not None:
                            on_token(i, token_id)
                # MOVE 3 — LATCH: OR in the EOS flags; once True, stays True.
                finished = finished | is_eos
                # MOVE 4 — EXIT: everyone done, nothing left worth computing.
                if finished.all().item():
                    break
                # Last iteration: the token for this step is already collected,
                # so another forward pass would produce a token nobody can use.
                if step == max_steps - 1:
                    break
                # MOVE 5 — GROW THE MASK: one new real (non-pad) position per
                # step, so append a column of 1s. dtype and device must match
                # the existing mask or the cat fails.
                #
                # HISTORY: the first working version was a RECOMPUTE loop — it
                # also grew input_ids by cat'ing the new token on and re-fed the
                # WHOLE sequence each step. Correct but O(n^2). Once its output
                # was verified, the KV cache was ported in: input_ids no longer
                # grows, only the single-token column is fed. The mask still has
                # to grow because it describes the full cached length, not just
                # the new token.
                token_col = next_token_ids.unsqueeze(1)          # [batch] -> [batch, 1]
                ones_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones_col],dim = 1)
                # MOVE 6 — FORWARD: new token column + full mask + cache, then
                # slice the last position and argmax back to flat [batch] for the
                # next iteration.
                outputs = self.model( input_ids=token_col, attention_mask=attention_mask,past_key_values=past_key_values,use_cache=True,)
                scores = outputs.logits[:, -1, :]
                next_token_ids = torch.argmax(scores, dim=-1)
                past_key_values = outputs.past_key_values


        # Raw bins out. Detokenization happens in generate_batch (outside
        # no_grad — it is pure tokenizer work, no tensors involved).
        return generated_tokens


    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: list[int],
        on_token: Callable[[int, int], None] | None = None,
    ) -> list[tuple[str, int]]:
        """Public batched entry point — what main.py's worker calls.

        Returns one (text, tokens_generated) tuple per prompt, in prompt order.

        The token count comes from len(bin), i.e. the ids BEFORE detokenization,
        because that is the true amount of model work done for that request.
        Counting after decode would undercount (skip_special_tokens drops
        tokens) and would not be comparable to the single-prompt path.

        on_token is passed straight through to _generate_batch_ids (see there).
        """
        bins = self._generate_batch_ids(prompts, max_new_tokens, on_token=on_token)
        return [
        (
            self.tokenizer.decode(tokens, skip_special_tokens=True),
            len(tokens),
        )
        for tokens in bins
    ]
