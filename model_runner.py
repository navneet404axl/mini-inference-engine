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
        raise NotImplementedError(
            "generate_tokens is yours to implement — see the STEP comments above."
        )
