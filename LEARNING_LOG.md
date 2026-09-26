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
