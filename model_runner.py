"""
model_runner.py

Loads Qwen2.5-1.5B from HuggingFace onto the GPU and exposes a small wrapper
for turning a text prompt into generated text.

Boundary (see CLAUDE.md):
  - Model loading + the tokenize/detokenize wrapper are boilerplate -> fully written here.
  - The autoregressive decode loop (generate_tokens) is left as a STUB for the
    human to implement. Everything else is wired to call it, so the server works
    end-to-end the moment that stub is filled in.
"""

from __future__ import annotations

import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Override via env vars without touching code.
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B")


def _pick_device(requested: str | None = None) -> torch.device:
    """Resolve which device to load onto. Prefer CUDA (the T4 target); fall
    back to CPU so the code still runs for local dev when no GPU is attached."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA not available — falling back to CPU. Generation will be slow.")
    return torch.device("cpu")


def _pick_dtype(device: torch.device) -> torch.dtype:
    """fp16 on GPU (T4 supports it; halves memory vs fp32), fp32 on CPU."""
    return torch.float16 if device.type == "cuda" else torch.float32


class ModelRunner:
    """Owns the tokenizer + model and runs generation.

    Construct once at server startup and reuse for every request — loading the
    weights is the expensive part and must not happen per-request.
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

        # Qwen ships an explicit eos token; fall back to it for padding so batched
        # work (week 3) has a defined pad id.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Model ready.")

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # Boilerplate wrapper: prompt -> token ids -> (your decode loop) ->   #
    # token ids -> text. The autoregressive loop in the middle is YOURS.  #
    # ------------------------------------------------------------------ #
    def generate_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int]:
        """Tokenize a prompt, run the decode loop, and detokenize the result.

        Returns (generated_text, num_tokens_generated). The generated text is
        ONLY the new tokens — the prompt is not echoed back.
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
    # STUB — YOU IMPLEMENT THIS (CLAUDE.md: the decode/token gen loop).   #
    # ------------------------------------------------------------------ #
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
    ) -> list[int]:
        """
        Run the autoregressive decode loop.

        Input:
          - input_ids:   LongTensor of shape (1, prompt_len) already on self.device
          - max_tokens:  max number of NEW tokens to generate
          - temperature: sampling temperature (0 or near-0 -> effectively greedy)
        Output:
          - list[int] of the NEW token ids only (do not include the prompt)

        Tools you have on hand:
          - self.model(input_ids=..., use_cache=True, past_key_values=...) -> outputs
              outputs.logits has shape (batch, seq_len, vocab)
              outputs.past_key_values is the KV cache to feed back next step
          - self.eos_token_id to know when to stop early
          - torch.no_grad() / torch.inference_mode() to skip autograd

        # STEP 1: set up the KV cache / initial state (your logic here)
        #   - run ONE forward pass on the full prompt to prime the cache
        #   - grab the past_key_values and the logits for the last position

        # STEP 2: loop up to max_tokens times (your logic here)
        #   - take logits[:, -1, :] (the next-token distribution)
        #   - apply temperature, softmax, then sample (or argmax if temp ~ 0)
        #   - record the sampled token id
        #   - break early if it equals self.eos_token_id
        #   - feed ONLY that new token back in with the cached past_key_values

        # STEP 3: return the list of generated token ids (your logic here)
        """
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values =  outputs.past_key_values
            generated = []                          # collect new token ids here

            for i in range(max_tokens):
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
                outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
        return generated

    # ------------------------------------------------------------------ #
    # STUB — YOU IMPLEMENT THIS (CLAUDE.md: the decode/token gen loop).   #
    # Streaming twin of generate_tokens: same loop, but YIELDS decoded    #
    # text per token instead of collecting ids into a list.              #
    # ------------------------------------------------------------------ #
    def stream_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
    ):
        """
        Autoregressive decode loop as a GENERATOR — yields text incrementally.

        Mirrors generate_tokens(), with two differences:
          1. Instead of appending token ids to a list and returning at the end,
             you YIELD the decoded text for each new token as it is produced.
          2. You stop BEFORE yielding the EOS token, so the caller never sees
             the end-of-sequence marker as output.

        Input:
          - input_ids:   LongTensor of shape (1, prompt_len) already on self.device
          - max_tokens:  max number of NEW tokens to generate
          - temperature: sampling temperature (0 or near-0 -> effectively greedy)
        Yields:
          - str: the decoded text for each new token, one at a time
                 (self.tokenizer.decode([token_id]))

        # STEP 1: prime the KV cache (your logic here)
        #   - run ONE forward pass on the full prompt with use_cache=True
        #   - keep outputs (for logits) and outputs.past_key_values (the cache)

        # STEP 2: loop up to max_tokens times (your logic here)
        #   - take logits[:, -1, :], apply temperature + softmax + sample,
        #     or argmax when temperature ~ 0  (same sampling as generate_tokens)
        #   - convert the sampled tensor to a plain int token_id
        #   - CHECK EOS FIRST: if token_id == self.eos_token_id, break and do
        #     NOT yield it (this is the key difference — check BEFORE yielding)
        #   - otherwise: yield self.tokenizer.decode([token_id])
        #   - feed ONLY the new token back in with past_key_values + use_cache=True,
        #     then refresh past_key_values from the new outputs

        # STEP 3: nothing to return — a generator just stops when the loop ends
        """
        with torch.no_grad():
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


                if token_id == self.eos_token_id:    # stop early
                    break
                yield self.tokenizer.decode([token_id])
                outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values 
                    
        # raise NotImplementedError(
        #     "stream_tokens is yours to implement — see the STEP comments above."
        # )
        # yield  # unreachable: makes this a generator function NOW, so the
        #        /generate/stream endpoint can iterate it. Delete this line
        #        (and the raise above) once your loop's own `yield` is in place.

    # ------------------------------------------------------------------ #
    # STUB — YOU IMPLEMENT THIS (CLAUDE.md: the batching logic).          #
    # ------------------------------------------------------------------ #
    def _generate_batch_ids(self, prompts: list[str], max_new_tokens: list[int]) -> list[list[int]]:
        # STEP 1: Configure tokenizer for batching — padding side + pad token
        # TODO Jul 5: move tokenizer config to __init__ — mutating shared state per-call is a smell
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # STEP 2: Tokenize all prompts into one padded batch (input_ids + attention_mask), move to device
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # STEP 3: Single batched forward pass (no grad)
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # STEP 4: Extract next-token logits for each sequence from the correct position
            scores = outputs.logits[:, -1, :]
            # STEP 5: Greedy pick — per-sequence first generated token (flat [batch] = home shape)
            next_token_ids = torch.argmax(scores, dim=-1)  # shape: [batch]

            # STEP 6: init bookkeeping for the decode loop
            batch_size = inputs["input_ids"].shape[0]
            generated_tokens = [[] for _ in range(batch_size)]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # STEP 7: batched decode loop — Recompute version (cache ported later, measured)
            # Home shape: flat [batch]; exactly ONE unsqueeze(1) at the cat.
            # DESIGN (locked Jul 7): Option A keep-and-ignore; EOS never collected;
            # flags = flip/filter/exit; mask grows +1s column per step.
     
            max_steps = max(max_new_tokens)
            for step in range(max_steps):

                # MOVE 1 — CHECK: compare this step's tokens (flat) against eos_token_id -> is_eos [batch] bool
                for i in range(batch_size):
                    if len(generated_tokens[i]) >= max_new_tokens[i]:
                        finished[i] = True
                is_eos = next_token_ids == self.eos_token_id    # bool [batch]
                # MOVE 2 — COLLECT: per sequence i: if not finished AND not EOS -> append token into bin i
                for i in range(batch_size):
                    if not finished[i] and not is_eos[i]:
                        generated_tokens[i].append(next_token_ids[i].item())
                # MOVE 3 — FLIP: OR is_eos into finished (once True, stays True)
                finished = finished | is_eos
                # MOVE 4 — EXIT: if all finished -> break
                if finished.all().item():
                    break
                if step == max_steps - 1:
                    break
                # MOVE 5 — GROW: unsqueeze tokens to [batch,1], cat onto input_ids (dim=1);
                #                cat ones-column onto attention_mask (dtype + device must match)
                # next_token_ids = next_token_ids.unsqueeze(1) dont do this not good 
                # input_ids = torch.cat([input_ids, next_token_ids],dim=1)
                token_col = next_token_ids.unsqueeze(1)          # costume, worn once
                input_ids = torch.cat([input_ids, token_col], dim=1)
                ones_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones_col],dim = 1)
                # MOVE 6 — FORWARD: full input_ids + full mask (recompute), slice [:, -1, :],
                #                   argmax dim=-1 (flat) -> next step's tokens
                outputs = self.model( input_ids=input_ids, attention_mask=attention_mask)
                scores = outputs.logits[:, -1, :]
                next_token_ids = torch.argmax(scores, dim=-1)

                
        
        # STEP 8: decode each bin to text, skip_special_tokens=True, return list[str]
        # (outside no_grad — tokenizer work, no gradients involved)

        return generated_tokens
# CHANGED Jul 9: was decode+return text here; now returns raw bins — split for verify_decode (option c), decode moved to public wrapper


    def generate_batch(self, prompts: list[str], max_new_tokens: list[int]) -> list[tuple[str, int]]:
        # RESERVED (Stage 4): decide return shape here — worker needs per-request
        # token counts (bins hold them); either return richer data or let worker
        # call _generate_batch_ids + decode separately.
        bins = self._generate_batch_ids(prompts,max_new_tokens)
        return [
        (
            self.tokenizer.decode(tokens, skip_special_tokens=True),
            len(tokens),
        )
        for tokens in bins
    ]
        