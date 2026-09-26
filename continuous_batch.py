"""
continuous_batch.py — iteration-level scheduling (continuous batching).

=============================================================================
WHAT THIS FILE IS
=============================================================================
A decode engine whose batch membership can change BETWEEN steps:

    add(requests)  prefill newcomers, sample their first token, merge them
                   into the running batch
    step()         one decode step for every running row; rows that finish
                   are evicted immediately
    evict(pred)    drop rows early (e.g. the client disconnected)

main.py's continuous worker drives it: admit waiting requests -> step ->
route tokens -> repeat. Like model_runner.py, it knows nothing about HTTP or
queues; rows are identified by an opaque `key` the caller supplies.

=============================================================================
WHY (vs the static batcher in model_runner.generate_batch)
=============================================================================
Static batching fixes the batch for its whole lifetime:
  - a 5-token request stuck with a 500-token one waits for all 500
    (head-of-line blocking),
  - a request arriving one step after a batch starts waits for the whole batch,
  - finished rows keep burning compute ("keep and ignore").
Here every step is a scheduling decision, so a finished row frees its slot
right away and a waiting request takes it on the next step. This is the core
idea of Orca (OSDI '22) and of vLLM / TGI's schedulers.

=============================================================================
THE HARD PART: ONE KV CACHE, ROWS OF DIFFERENT AGES
=============================================================================
The cache is per layer: keys/values of shape [batch, heads, seq, head_dim].
Rows are LEFT-padded so every row's newest token sits in the last column
(same trick as the static path). Membership changes are done by editing
those tensors only when membership changes, not on every step:

  LEAVE  -> index_select the surviving rows on the batch dim
            (DynamicCache.batch_select_indices), then TRIM leading columns
            that are padding for every remaining row, so the cache shrinks
            back instead of growing forever.
  JOIN   -> prefill the newcomers as their own small batch, left-pad both
            caches to the same length, concat on the batch dim.

POSITION IDS (the subtle bug this design has to avoid)
  Keys in the cache already have RoPE applied at the position they were
  computed at. A newcomer is prefilled on its own, so its keys use positions
  0..p-1. After merging into a longer cache (L columns), the model's DEFAULT
  position for its next token would be L (it counts columns), which is a
  false gap of L - p between the newcomer's query and its own keys. That
  corrupts attention. So every row carries its OWN position counter (its
  count of real tokens) and we pass position_ids explicitly on every forward.
  Pad columns never matter: they are masked out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from transformers import DynamicCache

# on_token(key, token_id): called once per COLLECTED token, on whatever thread
# is running the engine (main.py runs it in an executor thread).
TokenHook = Callable[[Any, int], None]

# (key, prompt, max_new_tokens, temperature)
NewRequest = tuple[Any, str, int, float]

# (key, generated_token_ids): a row that left the batch
Finished = tuple[Any, list[int]]


def sample_next(logits: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
    """Per-row sampling: [batch, vocab] logits + [batch] temperatures -> [batch] ids.

    temperature == 0 means argmax: exact and deterministic, which is what
    lets verify_continuous.py compare against the single-sequence path.
    temperature > 0 means softmax(logits / T), then a multinomial draw.
    Both are computed for the whole batch and torch.where picks per row; that
    costs less than splitting the batch. The all-greedy case skips the
    softmax entirely.
    """
    greedy = torch.argmax(logits, dim=-1)
    sampled_rows = temps > 0
    if not bool(sampled_rows.any()):
        return greedy
    t = temps.clamp(min=1e-6).unsqueeze(1)                # avoid /0 on greedy rows
    probs = torch.softmax(logits.float() / t, dim=-1)     # float32: fp16 softmax can underflow
    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
    return torch.where(sampled_rows, sampled, greedy)


def _left_pad_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    return F.pad(mask, (length - mask.shape[1], 0), value=0)


def _left_pad_kv(x: torch.Tensor, length: int) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]; pad the seq dim (2nd pair in F.pad order)
    return F.pad(x, (0, 0, length - x.shape[2], 0))


@dataclass
class _Row:
    key: Any
    max_new_tokens: int
    generated: list[int] = field(default_factory=list)


class ContinuousBatch:
    """The running batch. All tensor state is index-aligned with self.rows.

    State (B = number of running rows, L = cached columns):
      cache        DynamicCache, per layer [B, heads, L, head_dim]
      mask         [B, L]  1 = real cached token, 0 = left padding
      next_tokens  [B]     token each row feeds on the next step: already
                           sampled and already collected, but not yet cached
      positions    [B]     RoPE position for that token = row's real length
      temps        [B]     per-row temperature
    """

    def __init__(self, runner) -> None:
        self.model = runner.model
        self.tokenizer = runner.tokenizer
        self.device = runner.device
        self.eos_token_id = runner.eos_token_id
        # Same config as the static path: left padding keeps each row's
        # newest token in the last column, so logits[:, -1] is valid for all.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Drop every row and all tensor state (used when the batch empties or fails)."""
        self.rows: list[_Row] = []
        self.cache: DynamicCache | None = None
        self.mask: torch.Tensor | None = None
        self.next_tokens: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.temps: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def keys(self) -> list[Any]:
        return [r.key for r in self.rows]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def add(self, requests: Sequence[NewRequest], on_token: TokenHook | None = None) -> list[Finished]:
        """Admit newcomers: prefill them together, collect first tokens, merge.

        Returns the newcomers that finished at once (EOS as the first token,
        or max_new_tokens == 1). They never enter the running batch.

        Cost note: this prefill runs while the running rows wait, so every
        admission briefly stalls decode for everyone else (a TPOT blip).
        Chunked prefill is the standard fix; it is out of scope here.
        """
        if not requests:
            return []

        enc = self.tokenizer([r[1] for r in requests], padding=True, return_tensors="pt").to(self.device)
        mask = enc["attention_mask"]
        # Real-token positions: 0..p-1 for each row, regardless of left padding.
        positions = (mask.cumsum(-1) - 1).clamp(min=0)

        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=mask,
            position_ids=positions,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        cache = out.past_key_values
        temps = torch.tensor([float(r[3]) for r in requests], device=self.device)
        first = sample_next(out.logits[:, -1, :], temps)

        rows = [_Row(key=k, max_new_tokens=m) for k, _, m, _ in requests]
        finished: list[Finished] = []
        keep = self._collect(rows, first.tolist(), on_token, finished)
        if not keep:
            return finished

        idx = torch.tensor(keep, device=self.device)
        if len(keep) < len(rows):
            cache.batch_select_indices(idx)
        self._merge(
            [rows[i] for i in keep],
            cache,
            mask[idx],
            first[idx],
            mask.sum(-1)[idx],      # next position = number of real prompt tokens
            temps[idx],
        )
        return finished

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, on_token: TokenHook | None = None) -> list[Finished]:
        """One decode step for every running row. Finished rows are evicted.

        Returns the rows that finished on this step.
        """
        if not self.rows:
            return []
        assert self.mask is not None and self.next_tokens is not None
        assert self.positions is not None and self.temps is not None

        # The token being fed becomes a real cached column for every row.
        ones = torch.ones((len(self.rows), 1), dtype=self.mask.dtype, device=self.device)
        self.mask = torch.cat([self.mask, ones], dim=1)

        out = self.model(
            input_ids=self.next_tokens.unsqueeze(1),
            attention_mask=self.mask,
            position_ids=self.positions.unsqueeze(1),
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = out.past_key_values
        tokens = sample_next(out.logits[:, -1, :], self.temps)
        self.next_tokens = tokens
        self.positions = self.positions + 1

        finished: list[Finished] = []
        # tolist(): ONE GPU->CPU sync per step for the whole batch, not per row.
        keep = self._collect(self.rows, tokens.tolist(), on_token, finished)
        if len(keep) < len(self.rows):
            self._select(keep)
        return finished

    # ------------------------------------------------------------------ #
    def evict(self, predicate: Callable[[Any], bool]) -> list[Finished]:
        """Remove rows whose key matches predicate (e.g. client disconnected)."""
        if not self.rows:
            return []
        gone = [(r.key, r.generated) for r in self.rows if predicate(r.key)]
        if gone:
            self._select([i for i, r in enumerate(self.rows) if not predicate(r.key)])
        return gone

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _collect(
        self,
        rows: list[_Row],
        token_ids: list[int],
        on_token: TokenHook | None,
        finished: list[Finished],
    ) -> list[int]:
        """Apply one sampled token per row. Returns the indices still running.

        Same termination rules as the static path, so outputs are comparable:
          - EOS: finished, and the EOS is NOT collected,
          - otherwise collect it (and fire on_token); the row is finished once
            it holds max_new_tokens.
        """
        keep: list[int] = []
        for i, (row, tid) in enumerate(zip(rows, token_ids)):
            if row.max_new_tokens <= 0 or tid == self.eos_token_id:
                finished.append((row.key, row.generated))
                continue
            row.generated.append(tid)
            if on_token is not None:
                on_token(row.key, tid)
            if len(row.generated) >= row.max_new_tokens:
                finished.append((row.key, row.generated))
                continue
            keep.append(i)
        return keep

    def _merge(self, rows, cache, mask, next_tokens, positions, temps) -> None:
        """JOIN: fold a prefilled group into the running batch."""
        if not self.rows:
            # Empty batch: adopt the newcomers' state as-is.
            self.rows, self.cache, self.mask = list(rows), cache, mask
            self.next_tokens, self.positions, self.temps = next_tokens, positions, temps
            self._trim()
            return

        assert self.cache is not None and self.mask is not None
        length = max(self.mask.shape[1], mask.shape[1])
        self.mask = torch.cat([_left_pad_mask(self.mask, length), _left_pad_mask(mask, length)], dim=0)
        for old, new in zip(self.cache.layers, cache.layers):
            old.keys = torch.cat([_left_pad_kv(old.keys, length), _left_pad_kv(new.keys, length)], dim=0)
            old.values = torch.cat([_left_pad_kv(old.values, length), _left_pad_kv(new.values, length)], dim=0)
        self.rows.extend(rows)
        self.next_tokens = torch.cat([self.next_tokens, next_tokens])
        self.positions = torch.cat([self.positions, positions])
        self.temps = torch.cat([self.temps, temps])

    def _select(self, keep: list[int]) -> None:
        """LEAVE: keep only rows at `keep` (in order), then trim padding."""
        if not keep:
            self.reset()
            return
        idx = torch.tensor(keep, device=self.device)
        assert self.cache is not None
        self.cache.batch_select_indices(idx)
        self.mask = self.mask[idx]
        self.next_tokens = self.next_tokens[idx]
        self.positions = self.positions[idx]
        self.temps = self.temps[idx]
        self.rows = [self.rows[i] for i in keep]
        self._trim()

    def _trim(self) -> None:
        """Drop leading columns that are padding for EVERY row.

        Without this, a long-running row that leaves would leave its columns
        behind as all-pad, and the cache would only ever grow. After a trim,
        L equals the longest remaining row's real length.
        """
        assert self.mask is not None and self.cache is not None
        real_cols = self.mask.any(dim=0)
        first_real = int(torch.argmax(real_cols.to(torch.int8)).item())
        if first_real == 0:
            return
        self.mask = self.mask[:, first_real:]
        for layer in self.cache.layers:
            layer.keys = layer.keys[:, :, first_real:, :]
            layer.values = layer.values[:, :, first_real:, :]
