"""
main.py — FastAPI server for the mini inference engine.

=============================================================================
WHAT THIS FILE IS
=============================================================================
The HTTP front door + the scheduler. It owns:
  - the two request shapes (Pydantic in/out models),
  - the queue that decouples HTTP handlers from the GPU,
  - the single background worker that batches requests and drives the model,
  - the latency/throughput metrics window.

It deliberately owns NO model code. Anything that touches weights lives in
model_runner.py; this file only decides WHEN and WITH WHOM a request runs.

=============================================================================
HOW WE GOT HERE (the shape of this file is the history of the project)
=============================================================================
v0 — one request, one thread.
    /generate was a plain `def` that called the model inline. Simple, correct,
    and hopeless under load: two concurrent clients meant two threads fighting
    over one GPU, and nothing bounded how many could pile up.

v1 — streaming (/generate/stream).
    Proved the decode loop could emit tokens incrementally over SSE. It
    talked to the model directly, bypassing the queue, because a single
    Future can't carry a token stream (fixed in v5).

v2 — producer/consumer (the big structural change).
    Every /generate call became a PRODUCER: it packs its prompt + params + a
    private Future into a GenRequest, drops it on REQUEST_QUEUE, and parks on
    the Future. One background worker() is the sole CONSUMER and the sole
    owner of the GPU. Flow:

        /generate (producer)                 worker() (single consumer)
            build GenRequest  ──┐          ┌──> get GenRequest
            put on queue        ├─ QUEUE ──┤    run the model
            await .future    <──┘          └──  future.set_result(...)

    Why bother, when the work is still serialized? Because funnelling every
    request through ONE place is the precondition for batching: a consumer
    that already holds the queue can choose to pull N items instead of 1.

v3 (2026-07-02) — dynamic batching.
    worker() grew a two-phase loop: a GRAB phase that assembles a batch under
    two limits (MAX_BATCH_SIZE, BATCH_WINDOW_S), then a RUN phase that spends
    ONE forward pass on the whole batch. This is where the throughput came
    from — the GPU cost per step is nearly flat in batch size, so 8 requests
    per pass is close to 8x the work for 1x the time.

v4 — metrics.
    Timestamps were threaded onto GenRequest as it travels, finished requests
    are recorded in a fixed-size window, and /metrics reports p50/p95/p99 +
    tokens/sec + queue depth. Percentiles, not averages, because tail latency
    is the number that actually describes a serving system.

v5 — streaming through the worker + TTFT/TPOT.
    The batched decode grew an on_token(i, token_id) hook. The worker runs the
    batch on a thread (asyncio.to_thread) so the event loop stays live, and
    the hook pushes each request's text into its own sink (an asyncio.Queue)
    via loop.call_soon_threadsafe. /generate/stream now shares the queue,
    backpressure and batching with /generate. The same hook stamps the first
    token, so /metrics reports TTFT and TPOT percentiles for both endpoints.

v6 — continuous batching (iteration-level scheduling).
    continuous_worker + continuous_batch.ContinuousBatch decide batch
    membership before EVERY decode step: finished rows leave immediately and
    waiting requests join on the next step, with per-row position_ids so a
    merged cache stays correct. Per-row temperature sampling, and
    disconnected streams are evicted. The static worker is kept behind
    SCHEDULER=static for side-by-side benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from continuous_batch import ContinuousBatch
from model_runner import ModelRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------- request / response models --------------------------- #
# The public API contract. Pydantic validates and coerces at the edge, so no
# handler below ever has to defend against a missing field or a temperature of
# 1e9 — the bounds (ge/le) are enforced before our code runs.
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to continue.")
    max_tokens: int = Field(64, ge=1, le=2048, description="Max NEW tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float


# --------------------------- metrics: completed-request record --------------------------- #
@dataclass(frozen=True)
class CompletedRequest:
    """One finished request, reduced to just what the metrics math needs.

    Frozen on purpose: once a request is done its record is history and must
    never be edited in place. Note this stores END-TO-END latency (arrival ->
    completion, i.e. queue wait + batch wait + GPU time), not model time —
    that is what a client actually experiences.

    t_completion is kept so compute_metrics() can derive a throughput window
    (newest - oldest) without a separate clock.
    """

    total_latency_ms: float
    tokens_generated: int
    t_completion: float
    # Stage 4 streaming metrics. Optional because they are not always defined:
    #   ttft_ms — arrival -> first token. None if the request produced no token.
    #   tpot_ms — average gap between tokens AFTER the first:
    #             (t_completion - t_first_token) / (tokens - 1).
    #             None for 0 or 1 tokens (no gap to measure).
    # TTFT is dominated by queue wait + batch window + prefill; TPOT by decode
    # step time (which grows with batch size). Separating them is what tells
    # you WHICH phase to optimize.
    ttft_ms: float | None = None
    tpot_ms: float | None = None


# --------------------------- the queue --------------------------- #
# The single hand-off point between the async HTTP handlers (producers) and the
# one background worker (consumer) that owns the GPU.
#
# Why BOUNDED, and why SMALL: maxsize caps how many requests can sit waiting in
# memory at once. Without a bound, an overloaded server accepts work forever,
# the backlog grows, every client's latency grows with it, and eventually the
# box OOMs — the classic failure where the server "never says no" and so fails
# for everyone at once. With a bound, the queue itself becomes the backpressure
# valve. 32 is a deliberately small placeholder: big enough to keep the worker
# fed, small enough that overload is felt immediately rather than hidden.
#
# (Constructing the Queue at import time is fine on Python 3.10+: it binds to
# the running loop lazily on first use, not here.)
REQUEST_QUEUE: "asyncio.Queue[GenRequest]" = asyncio.Queue(maxsize=32)

# Sliding window of completed requests; maxlen makes old records fall off the
# left automatically, so memory is O(1) and the numbers describe RECENT
# behaviour instead of averaging in the cold-start requests from an hour ago.
METRICS_WINDOW: "deque[CompletedRequest]" = deque(maxlen=1000)

# The two dynamic-batching knobs (added 2026-07-02, when the worker stopped
# consuming one request at a time). They encode the central tradeoff:
#   MAX_BATCH_SIZE  — throughput ceiling. Bigger batch = more requests served
#                     per forward pass, but more memory and a longer step.
#   BATCH_WINDOW_S  — latency ceiling on batch formation. How long a lone early
#                     request is willing to wait for company. Too big and a
#                     quiet server adds pure dead time to every request; too
#                     small and bursts never get to batch at all.
MAX_BATCH_SIZE = 8  # max requests per forward pass (both schedulers)
BATCH_WINDOW_S = 0.05  # static only: accumulation deadline, clock starts at first arrival

# Which scheduler drives the GPU (Stage 5). Both are kept so they can be
# benchmarked against each other on the same codebase:
#   "continuous" — iteration-level scheduling (continuous_worker). Default.
#   "static"     — the original fixed batches (static_worker).
# Continuous needs no BATCH_WINDOW_S: nobody waits for a batch to "fill";
# a request joins the running batch on the very next decode step.
SCHEDULER = os.getenv("SCHEDULER", "continuous")


# --------------------------- request container --------------------------- #
@dataclass
class GenRequest:
    """
    One in-flight generation request as it travels through the queue.

    The producer (/generate) builds one of these, drops it on REQUEST_QUEUE,
    then awaits `.future`. The worker pulls it off, runs the model, and fulfils
    `.future` with the result — that round trip is the whole point of the queue.

    It doubles as the metrics carrier: rather than a side table keyed by
    request id, each request simply carries its own timestamps along with it.
    """

    prompt: str
    max_tokens: int
    temperature: float
    # asyncio.Future = a one-shot "the result will arrive later" box. The
    # producer CREATES it and awaits it; the worker fills it via .set_result()
    # (or .set_exception() on failure), which is what wakes the awaiting
    # producer. It must be created on the running loop, so the producer makes
    # it and passes it in here — one fresh Future per request, so results can
    # never cross wires between clients.
    #
    # Stage 4: a request now carries EXACTLY ONE of two reply channels:
    #   future — /generate: one value, delivered once at the end.
    #   sink   — /generate/stream: many values (text chunks), then a None
    #            end-of-stream sentinel. `loop` is the event loop that owns the
    #            sink; the worker's executor thread needs it for
    #            loop.call_soon_threadsafe(), because asyncio.Queue is NOT
    #            thread-safe and must only be touched from its own loop.
    future: "asyncio.Future[GenerateResponse] | None" = None
    sink: "asyncio.Queue[str | None] | None" = None
    loop: asyncio.AbstractEventLoop | None = None

    # --- metrics timestamps (monotonic clock; optional so each stage can stamp
    #     only the one it owns) ---
    t_arrival: float | None = None       # stamped by /generate the moment the request is born
    t_batch_entry: float | None = None   # stamped by worker() when this request is pulled into a batch
    t_first_token: float | None = None   # stamped by route_token (worker thread) on this
                                         # request's first collected token — feeds TTFT
    t_completion: float | None = None    # stamped by complete_request() just before the reply goes out

    # Reading these tells you WHERE time went, which is the whole reason they
    # exist as separate fields:
    #   t_batch_entry - t_arrival     = time spent queued (server is saturated)
    #   t_completion  - t_batch_entry = batch wait + GPU time (model is slow)

    # --- lifecycle flags ---
    # done: the reply channel has been closed (result, sentinel or error sent).
    #   Guards against double-completion on partial failures.
    # cancelled: set by the stream handler when its client disconnects. The
    #   continuous worker evicts the row on the next step; the static worker
    #   can't (a static batch can't drop a row mid-flight) and just ignores it.
    done: bool = False
    cancelled: bool = False
    # Per-request incremental detokenizer, created on the first streamed token.
    decoder: "StreamDecoder | None" = None


# --------------------------- stream routing --------------------------- #
class StreamDecoder:
    """Incremental detokenizer for ONE request (the prefix/read-offset trick, as in HF TGI).

    Decoding one token in isolation is wrong for byte-level BPE: a single
    character (emoji, accented letter, CJK) can span several tokens, and some
    tokens only get their leading space right in context. So we keep:
        ids          — every id generated so far
        prefix_off   — start of a small context window already sent
        read_off     — end of what has been sent
    Each token: decode ids[prefix_off:read_off] and ids[prefix_off:], and emit
    only the part the second has beyond the first. If that ends in U+FFFD the
    character is still incomplete, so hold it back until the next token
    finishes it. Cost stays O(window) per token instead of re-decoding the
    whole sequence (which would be O(n^2) over a long generation).

    Lives on the request (not keyed by batch row) because under continuous
    batching a request's row index changes as other rows leave.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.prefix_off = 0
        self.read_off = 0

    def push(self, token_id: int) -> str | None:
        """Add one token; return the newly completed text, or None if nothing is ready."""
        self.ids.append(token_id)
        prefix_text = self.tokenizer.decode(self.ids[self.prefix_off:self.read_off], skip_special_tokens=True)
        new_text = self.tokenizer.decode(self.ids[self.prefix_off:], skip_special_tokens=True)
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            self.prefix_off = self.read_off
            self.read_off = len(self.ids)
            return new_text[len(prefix_text):]
        return None


def route_token(req: GenRequest, token_id: int, tokenizer) -> None:
    """Handle one collected token for one request. Runs on the WORKER THREAD.

    Shared by both schedulers. Must be cheap (it runs inside the decode loop)
    and must never touch an asyncio object directly, only via
    req.loop.call_soon_threadsafe(...).
    """
    # TTFT stamp for EVERY request, streamed or not, on the same monotonic
    # clock as t_arrival. A plain attribute write from this thread is safe: the
    # event loop only reads it after the batch/step await returns.
    if req.t_first_token is None:
        req.t_first_token = time.monotonic()

    # Non-stream requests stop here: their text is decoded once, at the end.
    # Cancelled streams stop here too: nobody is reading the sink.
    if req.sink is None or req.cancelled:
        return

    if req.decoder is None:
        req.decoder = StreamDecoder(tokenizer)
    chunk = req.decoder.push(token_id)
    if chunk:
        # asyncio.Queue is not thread-safe: schedule the put ON the loop that
        # owns the sink. call_soon_threadsafe also wakes that loop.
        assert req.loop is not None
        req.loop.call_soon_threadsafe(req.sink.put_nowait, chunk)


def make_on_token(batch: list[GenRequest], runner: ModelRunner) -> Callable[[int, int], None]:
    """Static path: adapt route_token to the runner's positional on_token(i, token_id).

    Row i of generate_batch == batch[i] (the same positional contract as its results).
    """
    tokenizer = runner.tokenizer

    def on_token(i: int, token_id: int) -> None:
        route_token(batch[i], token_id, tokenizer)

    return on_token


def complete_request(req: GenRequest, text: str, tokens_generated: int) -> None:
    """Record metrics, then close the request's reply channel. Runs ON the event loop.

    Shared by both schedulers. Bookkeeping BEFORE announcement: set_result()
    can schedule the waiting handler immediately, so the metrics must already
    be recorded.

    ORDERING (streams): can the None sentinel overtake a token? No. Every
    token put was scheduled with call_soon_threadsafe BEFORE the thread's call
    returned, and to_thread delivers its result to the worker coroutine through
    the same loop's FIFO callback queue, so all token puts run before the
    worker even resumes and gets here.
    """
    req.t_completion = time.monotonic()
    assert req.t_arrival is not None
    total_latency_ms = (req.t_completion - req.t_arrival) * 1000.0
    ttft_ms = None
    tpot_ms = None
    if req.t_first_token is not None:
        ttft_ms = (req.t_first_token - req.t_arrival) * 1000.0
        if tokens_generated >= 2:
            tpot_ms = (req.t_completion - req.t_first_token) * 1000.0 / (tokens_generated - 1)
    METRICS_WINDOW.append(CompletedRequest(
        total_latency_ms=total_latency_ms,
        tokens_generated=tokens_generated,
        t_completion=req.t_completion,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
    ))

    # Route the ending by reply channel:
    #   future -> the one value. latency_ms is overwritten by the producer with
    #             its own end-to-end number; this one is the same measure.
    #   sink   -> text already went out token by token; send the None sentinel.
    if req.sink is not None:
        req.sink.put_nowait(None)
    else:
        assert req.future is not None
        req.future.set_result(GenerateResponse(
            text=text, tokens_generated=tokens_generated, latency_ms=total_latency_ms,
        ))
    req.done = True


def fail_request(req: GenRequest, exc: Exception) -> None:
    """ZERO ORPHANS: close a request's channel with an error. Runs ON the event loop.

    A Future gets set_exception (FastAPI turns it into a 500). A stream gets
    the exception object as its last item; the handler turns it into an SSE
    error frame. The done flag skips requests already completed before the
    failure (partial success), because set_result twice would raise.
    """
    if req.done:
        return
    if req.sink is not None:
        req.sink.put_nowait(exc)
    elif req.future is not None and not req.future.done():
        req.future.set_exception(exc)
    req.done = True


# --------------------------- worker: static batching --------------------------- #
async def static_worker(runner: ModelRunner) -> None:
    """
    SCHEDULER=static. The single background consumer. Owns the GPU: it is the ONLY thing that
    calls the model on the /generate path, so all requests are serialized
    through here and nothing can race for the device. Runs forever until the
    shutdown hook cancels it.

    Input:
      - runner: the shared ModelRunner (weights loaded once at startup).
    Returns:
      - never returns normally; exits only via cancellation at an await point.

    HISTORY
      v2: one request at a time — get -> run -> resolve.
      v3 (2026-07-02): split into two phases so a single forward pass could be
      amortized over up to MAX_BATCH_SIZE requests.

    Each trip around the loop:
      STEP A — GRAB: assemble a batch off REQUEST_QUEUE, bounded by
        MAX_BATCH_SIZE and a BATCH_WINDOW_S deadline that starts at the FIRST
        arrival (the deadline does not reset as more requests arrive — a
        resetting window could be held open indefinitely by steady traffic).
      STEP B — RUN: one batched call into the runner, then resolve every
        Future in the batch.
    """
    while True:
        # ---------------- STEP A: GRAB PHASE ----------------
        # Block until there is at least one request. This is the idle state of
        # the whole server, and the await point where cancellation lands at
        # shutdown. Nothing below runs on an empty queue — we never spin.
        first_req = await REQUEST_QUEUE.get()
        first_req.t_batch_entry = time.monotonic()
        batch = [first_req]

        # The window opens NOW, on the first arrival, and is fixed from here.
        deadline = time.monotonic() + BATCH_WINDOW_S

        # Collect companions until the batch is full or the window closes.
        while len(batch) < MAX_BATCH_SIZE:
            # Each wait is bounded by the time LEFT on the original deadline,
            # not by a fresh BATCH_WINDOW_S — that is what keeps the total
            # added latency capped at BATCH_WINDOW_S no matter how many
            # requests trickle in.
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            try:
                req = await asyncio.wait_for(
                    REQUEST_QUEUE.get(),
                    timeout=remaining
                )
                req.t_batch_entry = time.monotonic()
                batch.append(req)

            # TimeoutError here is the GO signal, not a failure: "window closed,
            # ship what we have."
            except asyncio.TimeoutError:
                break

        # ---------------- STEP B: RUN PHASE ----------------
        # Two parallel lists, built in batch order, because the runner's batched
        # API is positional.
        #
        # INVARIANT: results[i] corresponds to batch[i] — generate_batch
        # preserves prompt order (verified: tokenizer + output bins are
        # positional, nothing sorts). If ANY reordering is ever introduced
        # (sort-by-length bucketing), this positional zip breaks SILENTLY and
        # clients get each other's text. (The continuous worker avoids this by
        # keying rows on the request object itself.)
        #
        # KNOWN LIMITATION: the static path is greedy. Per-request temperature
        # is accepted by the API but not applied here. Kept that way on
        # purpose: this path is the verified baseline for benchmarks. The
        # continuous path applies temperature per row.
        prompts = [req.prompt for req in batch]
        max_new_tokens = [req.max_tokens for req in batch]
        try:
            # The hook is ALWAYS passed (not just when a sink is present) so
            # /generate requests get a TTFT stamp too.
            on_token = make_on_token(batch, runner)

            # OFF THE EVENT LOOP (Stage 4). generate_batch is synchronous and
            # takes seconds; called inline it froze the whole event loop for
            # the entire batch — no SSE frame could flush, /health hung, and
            # new requests couldn't even reach put_nowait (so the 503 policy
            # never fired mid-batch; clients just stalled at the socket).
            # to_thread runs it on a worker thread while this coroutine
            # awaits. The GPU is still driven by exactly one caller — this
            # worker awaits the batch before grabbing the next — so nothing
            # races for the device; we only freed the loop.
            results = await asyncio.to_thread(
                runner.generate_batch,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                on_token=on_token,
            )
            # Count-in == count-out tripwire: a mismatch means the runner
            # dropped or duplicated a sequence, and zip() would silently
            # truncate rather than tell us.
            assert len(results) == len(batch)
            for req, (text, tokens_generated) in zip(batch, results):
                complete_request(req, text, tokens_generated)
        except Exception as e:
            # Batch-level failure policy: ZERO ORPHANS. If the shared call
            # blows up, every request in the batch must be closed, or those
            # clients hang forever. fail_request skips any already completed.
            logger.exception("batch of %d failed", len(batch))
            for req in batch:
                fail_request(req, e)
        finally:
            # One task_done() per get(), success or failure — the queue's
            # internal counter must balance regardless of what happened.
            for _ in batch:
                REQUEST_QUEUE.task_done()


# --------------------------- worker: continuous batching --------------------------- #
async def continuous_worker(runner: ModelRunner) -> None:
    """
    SCHEDULER=continuous (default). Iteration-level scheduling: batch
    membership is decided before EVERY decode step, not once per batch.

    Each trip around the loop is ONE decode step:
      1. ADMIT   — pull waiting requests into free slots (up to MAX_BATCH_SIZE).
                   Idle (nothing running) -> block on the queue; busy -> only
                   take what's already waiting (get_nowait), never wait for more.
      2. EVICT   — drop rows whose stream client disconnected.
      3. PREFILL — newcomers are prefilled and merged (engine.add), off-loop.
      4. STEP    — one decode step for every running row (engine.step), off-loop.
      5. RETIRE  — finished rows complete their requests; their slots are free
                   for the next trip's ADMIT.

    Compared to static_worker:
      - no batching window: a request joins on the next step (~one step of wait),
      - a short request never waits for a long one to finish,
      - disconnected streams stop costing compute right away.
    Cost: each admission's prefill pauses decode for the running rows (a TPOT
    blip). Chunked prefill is the standard fix; out of scope here.

    Rows are keyed by the GenRequest object itself, so there is no positional
    zip to break as rows come and go.
    """
    engine = ContinuousBatch(runner)
    tokenizer = runner.tokenizer

    def on_token(req: GenRequest, token_id: int) -> None:
        route_token(req, token_id, tokenizer)

    def retire(req: GenRequest) -> None:
        # One task_done() per get(), however the request ends.
        REQUEST_QUEUE.task_done()

    while True:
        # ---- 1. ADMIT ----
        new: list[GenRequest] = []
        if len(engine) == 0:
            # Idle state of the whole server, and where shutdown's
            # cancellation lands. Never spin on an empty queue.
            new.append(await REQUEST_QUEUE.get())
        while len(engine) + len(new) < MAX_BATCH_SIZE:
            try:
                new.append(REQUEST_QUEUE.get_nowait())
            except asyncio.QueueEmpty:
                break
        now = time.monotonic()
        for req in new:
            req.t_batch_entry = now

        # ---- 2. EVICT disconnected clients ----
        # Before paying for a newcomer's prefill or another step for a row
        # nobody is reading.
        live_new = []
        for req in new:
            if req.cancelled:
                req.done = True
                retire(req)
            else:
                live_new.append(req)
        evicted = engine.evict(lambda r: r.cancelled)
        for req, _ in evicted:
            req.done = True
            retire(req)
        if evicted:
            logger.info("evicted %d disconnected stream(s)", len(evicted))

        # Snapshot of everyone in flight, for the failure path: if either call
        # below raises, all of them must be closed (zero orphans).
        in_flight = engine.keys + live_new
        try:
            finished = []
            # ---- 3. PREFILL newcomers ----
            if live_new:
                finished += await asyncio.to_thread(
                    engine.add,
                    [(r, r.prompt, r.max_tokens, r.temperature) for r in live_new],
                    on_token,
                )
            # ---- 4. STEP ----
            if len(engine):
                finished += await asyncio.to_thread(engine.step, on_token)
        except Exception as e:
            logger.exception("continuous step failed; failing %d requests", len(in_flight))
            for req in in_flight:
                if not req.done:
                    fail_request(req, e)
                    retire(req)
            engine.reset()
            continue

        # ---- 5. RETIRE finished rows ----
        for req, ids in finished:
            complete_request(req, tokenizer.decode(ids, skip_special_tokens=True), len(ids))
            retire(req)


# --------------------------- app lifecycle --------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks.

    Two things must be process-wide singletons, and this is where they are
    created: the model (loading weights is seconds-expensive and must never
    happen per request) and the worker task (a second worker would mean two
    things driving one GPU, which is the exact problem the queue exists to
    solve).
    """
    # --- startup ---
    logger.info("Starting up — loading model ...")
    app.state.runner = ModelRunner()  # load weights once; reused for every request
    # create_task() SCHEDULES worker() to run concurrently on the event loop and
    # returns IMMEDIATELY with a Task handle — it does not block or run the
    # worker inline here. The handle is stashed on app.state so the shutdown
    # half below can cancel it.
    worker_fn = {"static": static_worker, "continuous": continuous_worker}[SCHEDULER]
    app.state.worker_task = asyncio.create_task(worker_fn(app.state.runner))
    logger.info("Background worker started (scheduler=%s).", SCHEDULER)

    yield  # <-- app serves requests for its whole lifetime here

    # --- shutdown ---
    logger.info("Shutting down — stopping worker ...")
    # cancel() only REQUESTS cancellation: it arranges for a CancelledError to
    # be raised inside the worker at its next await point (in practice the
    # `await REQUEST_QUEUE.get()` it idles on).
    app.state.worker_task.cancel()
    try:
        # Await the cancelled task so we actually wait for it to unwind before
        # the process exits. The CancelledError we just triggered propagates
        # out of this await — catching and ignoring it is the normal, clean way
        # a cancelled task is reaped. It is NOT an error here; it is the
        # expected exit signal.
        await app.state.worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Worker stopped.")


app = FastAPI(title="Mini LLM Inference Engine", version="2.0", lifespan=lifespan)


# --------------------------- routes --------------------------- #
@app.get("/health")
def health(request: Request) -> dict:
    """Liveness + "which model am I actually serving" check.

    Sync `def` and deliberately trivial: it must answer even when the GPU is
    saturated, so it never touches the queue or the model.
    """
    runner: ModelRunner = request.app.state.runner
    return {
        "status": "ok",
        "model_id": runner.model_id,
        "device": str(runner.device),
        "scheduler": SCHEDULER,
    }


# --------------------------- producer handler --------------------------- #
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """
    Producer side of the queue.

    `async def` (it was a sync `def` in v0): it runs ON the event loop, which
    is what lets it await the Future without occupying a thread. Hundreds of
    these can be parked at once for the cost of hundreds of small objects.
    It does NOT touch the model — it hands work to the worker and waits.

    Input:  the validated GenerateRequest (prompt, max_tokens, temperature).
    Output: the GenerateResponse the worker produced for THIS request.

    Steps: create a private Future -> wrap it with the params in a GenRequest
    (stamping arrival time) -> enqueue -> await -> patch in true end-to-end
    latency -> return.
    """
    future = asyncio.get_running_loop().create_future()
    # t_arrival is stamped here, at the earliest point we control, so queue
    # wait time is included in the latency we report. Stamping it in the worker
    # would flatter the numbers by hiding exactly the delay we care about.
    gen_req = GenRequest(
        prompt = req.prompt,
        max_tokens = req.max_tokens,
        temperature = req.temperature,
        future = future,
        t_arrival=time.monotonic()
    )

    try:
        # BACKPRESSURE POLICY (the decision this endpoint really makes):
        # put_nowait + reject, rather than `await put()` + wait.
        # `await REQUEST_QUEUE.put(...)` would block this handler until a slot
        # frees — the client sees an ever-growing timeout with no signal. Fail
        # fast with 503 instead: the client learns immediately that the server
        # is full and can retry or shed load. "No" now beats "maybe" later.
        REQUEST_QUEUE.put_nowait(gen_req) #no await needed if you were doing .put() the you would have needed await
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="server overloaded, try again later")
    # Suspends here until the worker resolves THIS Future. If the worker called
    # set_exception (batch failure), that exception is re-raised at this line
    # and FastAPI turns it into a 500.
    response = await gen_req.future
    # The worker filled latency_ms with the shared model time for the batch.
    # Overwrite it with this request's real end-to-end latency: queue wait +
    # batch wait + GPU. That is the number the caller experienced.
    assert gen_req.t_completion is not None
    assert gen_req.t_arrival is not None
    response.latency_ms = (gen_req.t_completion - gen_req.t_arrival) * 1000
    return response


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest, request: Request) -> StreamingResponse:
    """Stream generated text token-by-token as Server-Sent Events (SSE).

    v5 rewrite (was v1). The old version called the runner DIRECTLY and bypassed
    the queue/worker because a one-shot Future can't carry an incremental stream.
    This version routes through the SAME queue + worker path as /generate, but
    the request carries a per-request "sink" (an asyncio.Queue) instead of a
    Future — a Future delivers one value, a stream delivers many.

    THE HAND-OFF (whose code is whose):
      - worker thread (asyncio.to_thread): make_on_token's callback PRODUCES
        text chunks into the sink with loop.call_soon_threadsafe(sink.put_nowait,
        chunk); the worker pushes a single None to mark end-of-stream.
      - THIS handler: CONSUMES the sink and frames each token onto the wire. It
        does no model work and no blocking work — hence `async def` (v1 was sync
        `def` because it drove the GPU inline; this one only awaits a Queue, so
        it belongs on the event loop where one thread serves many live streams).

    WIRE FORMAT (SSE) — this is the contract test_stream.py parses:
      data: {"text": "<chunk>"}\n\n      <- one frame per token
      ...
      data: [DONE]\n\n                    <- terminal sentinel (sink yielded None)
    On batch failure, a single  data: {"error": "..."}  frame precedes [DONE].

    NOTE: the batched path is greedy, so `temperature` is accepted but not
    applied here (it was when this endpoint drove stream_tokens directly).

    Each chunk is JSON-encoded (not raw text) so a token containing newlines or
    quotes can't corrupt the SSE framing — a raw newline would end the frame
    early. `json.dumps` is already imported at the top of this file.
    """
    # THE SINK: this request's private token channel. Unbounded on purpose —
    # the producer is the worker thread, and it must NEVER block on a slow
    # client (that would stall the whole batch for everyone). Its size is
    # already bounded by max_tokens. The loop is captured so the worker thread
    # can schedule puts onto it with call_soon_threadsafe.
    loop = asyncio.get_running_loop()
    sink: "asyncio.Queue[str | Exception | None]" = asyncio.Queue()

    gen_req = GenRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        sink=sink,
        loop=loop,
        t_arrival=time.monotonic(),
    )

    # Same backpressure policy as /generate. This runs BEFORE the
    # StreamingResponse is returned, so a full queue is still a clean 503 —
    # once streaming starts, the 200 status has already been sent.
    try:
        REQUEST_QUEUE.put_nowait(gen_req)
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="server overloaded, try again later")

    async def token_stream():
        """Async generator drained by StreamingResponse on the event loop.

        Pull chunks off the sink until the None sentinel, framing each as an
        SSE 'data:' line, then emit [DONE]. An Exception in the sink means the
        batch failed: send an error frame, then [DONE], so the client always
        sees a terminated stream.

        CLIENT DISCONNECT: Starlette stops iterating and closes this
        generator; the finally below sets gen_req.cancelled. The continuous
        worker then evicts the row on its next step. The static worker can't
        drop a row mid-batch, so there the row runs to max_tokens (route_token
        stops pushing text for it either way).
        """
        finished = False
        try:
            while True:
                item = await sink.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    yield f"data: {json.dumps({'error': str(item) or type(item).__name__})}\n\n"
                    break
                yield f"data: {json.dumps({'text': item})}\n\n"
            yield "data: [DONE]\n\n"
            finished = True
        finally:
            # Client went away mid-stream (Starlette closed this generator).
            # Flag it so the continuous worker evicts the row next step.
            if not finished:
                gen_req.cancelled = True

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",   # SSE: the browser/client reads discrete events
        headers={
            "Cache-Control": "no-cache",       # never cache a live stream
            "Connection": "keep-alive",        # hold the socket open for the whole stream
            "X-Accel-Buffering": "no",         # tell nginx/proxies NOT to buffer -> tokens flush live
        },
    )


# --------------------------- metrics math --------------------------- #
def percentile(sorted_values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. Pure math, no globals, no I/O.

    Input:  values ALREADY sorted ascending (the caller sorts once and calls
            this three times — sorting inside would be 3x the work), and p as
            a fraction (0.95, not 95).
    Output: the value at that rank, or None for an empty list.

    Index convention: min(int(p * N), N - 1). The floor picks the nearest rank
    below; the min() clamp is what stops p=0.99 on a small window from indexing
    off the end (0.99 * 100 == 100, but the last valid index is 99).
    """
    ...
    n = len(sorted_values)

    if n == 0:
        return None

    idx = min(int(p * n), n - 1)

    return sorted_values[idx]

def compute_metrics() -> dict:
    """Reduce METRICS_WINDOW to the numbers /metrics reports.

    Single pass over the window to collect latencies, total tokens, and the
    oldest/newest completion times; then ONE sort feeding all three
    percentiles.

    Why percentiles and not a mean: an average hides the tail, and the tail is
    the user-visible failure mode — p99 is the request that queued behind a
    full batch. p50/p95/p99 together show the shape of that distribution.

    Throughput is derived from the window's own span (newest - oldest
    completion) rather than wall-clock uptime, so an idle period doesn't drag
    the number down. It needs at least 2 records to have a span at all, hence
    the None below; the span <= 0 guard covers the case where a whole batch
    completes inside one clock tick.

    Returns: p50_ms, p95_ms, p99_ms, sample_count, tokens_per_sec, plus
    ttft_p50/p95/p99_ms and tpot_p50/p95/p99_ms (Stage 4). TTFT/TPOT skip
    records where they're undefined, so they can be None even when latency
    isn't (e.g. a window of only 1-token requests has no TPOT).
    Everything except sample_count can be None when the window is too thin —
    the caller must treat "not enough data yet" as a real state, not zero.
    """
    ...
    latencies: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    total_tokens = 0
    oldest_t_completion: float | None = None
    newest_t_completion: float | None = None

    # One walk, three quantities. The deque is in completion order, so the
    # first record seen is the oldest and the last one assigned is the newest.
    for record in METRICS_WINDOW:
        latencies.append(record.total_latency_ms)
        total_tokens += record.tokens_generated
        if record.ttft_ms is not None:
            ttfts.append(record.ttft_ms)
        if record.tpot_ms is not None:
            tpots.append(record.tpot_ms)

        if oldest_t_completion is None:
            oldest_t_completion = record.t_completion

        newest_t_completion = record.t_completion

    sample_count = len(latencies)

    # Empty window: report honestly rather than inventing zeros, which would
    # read as "0ms latency, server is perfect".
    if sample_count == 0:
        return {
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "sample_count": 0,
            "tokens_per_sec": None,
            "ttft_p50_ms": None,
            "ttft_p95_ms": None,
            "ttft_p99_ms": None,
            "tpot_p50_ms": None,
            "tpot_p95_ms": None,
            "tpot_p99_ms": None,
        }

    latencies.sort()

    p50_ms = percentile(latencies, 0.50)
    p95_ms = percentile(latencies, 0.95)
    p99_ms = percentile(latencies, 0.99)

    # percentile() already returns None for an empty list, so "no record has a
    # TTFT yet" falls out as None with no special case.
    ttfts.sort()
    tpots.sort()

    if sample_count < 2:
        tokens_per_sec = None
    else:
        assert oldest_t_completion is not None
        assert newest_t_completion is not None

        span = newest_t_completion - oldest_t_completion

        if span <= 0:
            tokens_per_sec = None
        else:
            tokens_per_sec = total_tokens / span

    return {
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "sample_count": sample_count,
        "tokens_per_sec": tokens_per_sec,
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "tpot_p50_ms": percentile(tpots, 0.50),
        "tpot_p95_ms": percentile(tpots, 0.95),
        "tpot_p99_ms": percentile(tpots, 0.99),
    }

@app.get("/metrics")
def metrics() -> dict:
    """Scrape endpoint.

    queue_depth is added here rather than inside compute_metrics() because it
    is a LIVE gauge (how backed up are we right now), while everything else is
    a summary of finished work. Read together they tell you the difference
    between "slow model" and "too much traffic".
    """
    out = compute_metrics()
    out["queue_depth"] = REQUEST_QUEUE.qsize()
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
