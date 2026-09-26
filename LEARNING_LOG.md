# Learning Log — code Claude wrote that I still need to learn

Each entry lists what was built, why it's built that way, the concept behind
it, and questions to check I really understand it. Goal: be able to rebuild
each piece from scratch and explain it in an interview.

---

## Stage 4 · Step 1 — Streaming through the worker + TTFT/TPOT (2026-09-26)

**Files:** `main.py` (worker, `make_on_token`, `/generate/stream`, metrics),
`model_runner.py` (`on_token` hook in `_generate_batch_ids`).

**Verified:** `test_stream.py` streams incrementally. Streamed text is
byte-identical to `/generate` for 4 prompts (incl. Chinese). `/health` stays
<25 ms during a batch. Failure path: stream gets its tokens then an error
frame, and no request hangs. `verify_decode` 4/4.

### 1. The event loop was being frozen (the bug under everything)
- **What changed:** `runner.generate_batch(...)` → `await asyncio.to_thread(runner.generate_batch, ...)`.
- **Why:** an `async def` that calls a slow *synchronous* function blocks the
  whole event loop. Nothing else runs: no SSE frames flush, `/health` hangs,
  new requests can't reach `put_nowait`, so the 503 policy never fired mid-batch.
- **Concept:** asyncio is cooperative. Only code that hits an `await` gives up
  the thread. CPU/GPU-bound work belongs in a thread (or a process).
- **Why it's still safe for the GPU:** one worker, and it awaits each batch
  before grabbing the next, so there's still exactly one caller of the model.
- **Check yourself:**
  - Why didn't this bug show up with plain `/generate`?
  - What's the difference between `asyncio.to_thread` and `run_in_executor`?
  - Would two workers + `to_thread` be OK? What breaks?

### 2. The `on_token(i, token_id)` hook in the batched decode
- **Where:** MOVE 2 COLLECT in `_generate_batch_ids`, right after the append.
- **Why there:** it's the only spot that knows row→token *and* has already
  filtered EOS and finished rows. The hook sees exactly the tokens that land
  in the output bins.
- **Detail:** reuses the int from `.item()`. A second `.item()` would be a
  second GPU→CPU sync per token.
- **Design choice (yours):** callback over generator. Smaller change, runner
  stays HTTP-agnostic, costs nothing when `None`.
- **Check yourself:** what would a generator version look like, and why is it
  closer to what continuous batching needs?

### 3. One request, two reply channels: Future vs sink
- `GenRequest` has `future` (one value, `/generate`) **or** `sink` + `loop`
  (many values then `None`, `/generate/stream`).
- **Why a Queue and not a Future:** a Future resolves once. A stream is many values.
- **Why the sink is unbounded:** the producer is the worker thread. If it ever
  blocked on one slow client, the whole batch would stall for everyone. Size
  is already bounded by `max_tokens`.

### 4. Thread → event loop hand-off: `call_soon_threadsafe`
- The callback runs on the worker *thread*. `asyncio.Queue` is not
  thread-safe, so the thread must never call `sink.put_nowait` directly.
- `loop.call_soon_threadsafe(sink.put_nowait, chunk)` schedules the put *on*
  the loop that owns the sink, and wakes that loop.
- **Ordering proof (why `None` can't overtake a token):** all token puts are
  scheduled before `generate_batch` returns. `to_thread` delivers its result
  back through the same loop's FIFO callback queue, so every put runs before
  the worker coroutine even resumes and pushes `None`.
- **Check yourself:** what exactly goes wrong if the thread calls
  `sink.put_nowait` directly? (Hint: waking the waiter, internal deque races.)

### 5. Incremental detokenization (`make_on_token`)
- **Problem:** Qwen uses byte-level BPE. One character (emoji, 中文) can span
  several tokens, so `decode([id])` alone gives `�` garbage. Leading
  spaces can also depend on context.
- **Fix (HF TGI's prefix/read-offset trick):** per row, keep `ids`,
  `prefix_off`, `read_off`. Decode `ids[prefix_off:read_off]` and
  `ids[prefix_off:]`, and emit only the extra suffix. If it ends in `�`,
  hold it back, because the next token completes the character.
- **Why not re-decode everything each token:** O(n²) over a long generation.
  The window keeps it O(small).
- **Check yourself:** walk through a 3-token emoji by hand. What's emitted after
  token 1, 2, 3?

### 6. Zero orphans for streams
- If the batch throws, every sink not yet closed gets the **exception object**
  as its last item. The handler turns it into `data: {"error": ...}` then `[DONE]`.
- The `closed` set tracks requests already finished before the failure
  (partial success), so they don't get a second terminator.
- **Check yourself:** why must the 503 check happen *before* returning
  `StreamingResponse`? (Hint: once streaming starts, the 200 status is already sent.)

### 7. TTFT and TPOT
- `t_first_token` is stamped in the callback for **every** request (streamed or
  not), on the same `time.monotonic()` clock as `t_arrival`.
- `TTFT = t_first_token − t_arrival`: queue wait + batch window + prefill.
- `TPOT = (t_completion − t_first_token) / (tokens − 1)`: the decode step time.
  Undefined for 0–1 tokens, so it's `None`.
- **Why both:** TTFT tells you about scheduling/prefill, TPOT about decode
  speed. They point at different fixes. Users *feel* TTFT for chat.
- **Check yourself:** in the CPU test, TPOT p50 was ~918 ms but test_stream
  showed ~300 ms gaps. Why? (Hint: batch size 8 vs 1.)

### Known limitations (deliberate, for now)
- The batched path is greedy: `temperature` is accepted but ignored.
- Client disconnect: the row keeps decoding until `max_tokens` (can't drop a row
  mid-batch). **Continuous batching fixes this.**
- Static batching: short requests wait for the longest in their batch.
  **Continuous batching fixes this too.** That's the next stage.

---

## Stage 5 — Continuous batching (2026-09-26)

**Files:** `continuous_batch.py` (new: the `ContinuousBatch` engine),
`main.py` (`continuous_worker`, shared helpers `route_token` /
`complete_request` / `fail_request`, `StreamDecoder`, `SCHEDULER` switch),
`verify_continuous.py` (new correctness test).

**Verified (CPU, fp32):**
- `verify_continuous.py`: 6 requests joining and leaving at staggered steps
  (join into empty batch, join a longer cache, join a shorter cache, finish
  inside prefill, leave-then-trim, join after trim). **All token IDs identical**
  to the single-sequence path.
- The same test with `position_ids` removed: rows A, B, C **fail**. That
  proves the position fix is necessary, not decoration.
- Head-of-line test (one 40-token request, then 3 × 4-token requests 1 s later):
  shorts took **14.7 s static vs 3.2 s continuous (4.6× faster)**.
- Mixed stream + non-stream: streamed text == `/generate` text on both schedulers.
- Disconnect: the stream is evicted on the next step (`evicted 1 disconnected stream(s)`).
- Temperature 1.2 gives 3 different outputs, temperature 0 gives identical ones.
- Crash mid-step: every in-flight request gets an error, the queue counter
  balances, and the worker survives.

### The idea in one paragraph
Static batching decides the batch **once**: it lives until its longest member
finishes. Continuous (iteration-level) batching decides **before every decode
step**: finished rows leave immediately, waiting requests join on the next
step. From Orca (OSDI '22); it's the core of vLLM/TGI schedulers. The win is
latency for short requests, better GPU occupancy, and no wasted rows.

### Decision 1 — KV cache: merge/evict on membership events ✅
- **Chosen:** one batched `DynamicCache`, left-padded. *Leave* =
  `batch_select_indices` (keep surviving rows) + **trim** leading all-pad
  columns. *Join* = prefill newcomers as their own mini-batch, left-pad both
  caches to equal length, `torch.cat` on the batch dim. Copy cost is paid
  only when membership changes, not every step. Works with stock HF.
- **Rejected: per-request caches re-stacked every step.** Simplest to reason
  about, but copies the whole cache every step, so throughput collapses as
  contexts grow.
- **Rejected (for now): paged KV cache (vLLM).** Fixed-size blocks + block
  tables, zero padding waste. It's the "real" answer but needs a custom
  attention kernel. It's the planned stretch goal.
- **Check yourself:** why left padding and not right? What does `_trim`
  prevent, and what would happen to memory without it?

### Decision 2 — explicit per-row `position_ids` (the bug I caught before it happened)
- Keys in the cache already have **RoPE baked in** at the position they were
  computed at. A newcomer prefilled alone has keys at 0..p−1. Merged into an L-column
  cache, the model's **default** position for its next token = L (it counts
  columns), a false gap of L−p. Attention quietly degrades.
- **Fix:** each row carries its own counter (= its count of real tokens), passed
  as `position_ids` on every forward. Prefill uses `mask.cumsum(-1) − 1`.
- **Why the static path never hit this:** everything was prefilled together,
  and RoPE only cares about *relative* distance. Shifting a whole row by its
  pad count changes nothing. The mismatch only appears when a row's keys and
  its next query were computed under different offsets, which is exactly a merge.
- **Check yourself:** explain why RoPE makes a constant shift harmless but a
  merge harmful. Which rows failed without the fix, and why those?

### Decision 3 — admission: every step, whenever slots are free ✅
- **Chosen:** before each step, pull everything already waiting into free
  slots (`get_nowait`, never wait for more). Block on the queue only when idle.
- **Rejected: admit every K steps / when M are waiting.** Fewer merges and
  smoother TPOT, but worse TTFT and two more knobs to tune.
- **Known cost:** a newcomer's prefill runs while running rows wait, which
  causes a TPOT blip for them. The standard fix is **chunked prefill** (split
  long prompts across steps). Good future chart, good interview topic.
- No `BATCH_WINDOW_S` in continuous mode: nobody waits for a batch to fill.

### Decision 4 — keep static behind `SCHEDULER=static` ✅
- Needed for the before/after benchmark on the same codebase (Stage 6).
- The static path stays greedy on purpose: it's the verified baseline.

### Decision 5 — per-row temperature sampling ✅
- `sample_next(logits, temps)`: argmax for the whole batch, softmax/multinomial
  for the whole batch, then `torch.where(temps > 0, ...)` picks per row. The
  all-greedy case skips the softmax entirely.
- The softmax runs in float32 because fp16 can underflow small probabilities.
- Temperature 0 stays exact argmax, which is what lets the test compare IDs.

### Decision 6 — rows keyed by the request object
- The engine stores an opaque `key` per row; the worker passes the
  `GenRequest` itself. No positional zip to break when rows leave and
  indices shift. (The static path's positional contract was a known landmine.)
- For the same reason `StreamDecoder` moved **onto the request**: a row's
  index changes as others leave, the request doesn't.

### Decision 7 — shared completion/failure helpers
- `complete_request` / `fail_request` are used by both workers, so metrics
  and zero-orphans logic exist once. `req.done` replaced the per-batch
  `closed` set, because in continuous mode requests finish at different times.
- Failure policy: snapshot everyone in flight *before* the risky calls. On
  exception, fail them all, `reset()` the engine, and keep serving.

### Decision 8 — disconnect eviction
- The stream handler's `finally` sets `req.cancelled` if the stream didn't
  finish. The continuous worker evicts such rows before the next step (and
  skips cancelled newcomers before paying for their prefill).
- Static can't do this (a fixed batch can't drop a row), which is another
  point for continuous.

### Check yourself (overall)
- Walk one request through the continuous worker loop: ADMIT → EVICT →
  PREFILL → STEP → RETIRE. Where can it wait, and for how long at most?
- Why is `tokens.tolist()` once per step better than `.item()` per row?
- What limits `MAX_BATCH_SIZE` now? (Hint: KV memory per row × context length.)
- What would paged attention change about `_merge` / `_select` / `_trim`?
