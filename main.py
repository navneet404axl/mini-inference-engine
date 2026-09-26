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
    Proved the decode loop could emit tokens incrementally over SSE. Still
    talked to the model directly — and it still does today (see the note on
    that endpoint), because a single Future can't carry a token stream.

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
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
MAX_BATCH_SIZE = 8  # max requests per forward pass
BATCH_WINDOW_S = 0.05  # accumulation deadline in seconds; clock starts at first arrival


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
    future: "asyncio.Future[GenerateResponse]"

    # --- metrics timestamps (monotonic clock; optional so each stage can stamp
    #     only the one it owns) ---
    t_arrival: float | None = None       # stamped by /generate the moment the request is born
    t_batch_entry: float | None = None   # stamped by worker() when this request is pulled into a batch
    t_first_token: float | None = None   # reserved for TTFT; nothing writes it yet — the batched
                                         # path has no per-request first-token hook (see model_runner)
    t_completion: float | None = None    # stamped by worker() just before set_result

    # Reading these tells you WHERE time went, which is the whole reason they
    # exist as separate fields:
    #   t_batch_entry - t_arrival     = time spent queued (server is saturated)
    #   t_completion  - t_batch_entry = batch wait + GPU time (model is slow)


# --------------------------- the worker --------------------------- #
async def worker(runner: ModelRunner) -> None:
    """
    The single background consumer. Owns the GPU: it is the ONLY thing that
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
        # (sort-by-length bucketing, continuous batching), this positional zip
        # breaks SILENTLY and clients get each other's text — add explicit id
        # plumbing BEFORE any such change.
        #
        # KNOWN LIMITATION (also in README): the batched path is greedy. Per-
        # request temperature is accepted by the API but not applied here.
        # Deliberate: correctness of batching was verified first; batched
        # sampling is future work.
        prompts = [req.prompt for req in batch]
        max_new_tokens = [req.max_tokens for req in batch]
        start = time.perf_counter()
        try:
            results = runner.generate_batch(prompts=prompts,max_new_tokens=max_new_tokens)
            # Model time for the batch as a whole. Every member of the batch
            # gets the same number here — it is the cost of the shared pass,
            # not a per-request measurement. The per-request end-to-end number
            # is computed by the producer from its own timestamps.
            model_latency_ms = (time.perf_counter() - start) * 1000.0
            # Count-in == count-out tripwire: a mismatch means the runner
            # dropped or duplicated a sequence, and zip() would silently
            # truncate rather than tell us.
            assert len(results) == len(batch)

            for req, (text, tokens_generated) in zip(batch, results):
                response = GenerateResponse(text=text,tokens_generated=tokens_generated,latency_ms=model_latency_ms)
                req.t_completion = time.monotonic()
                assert req.t_arrival is not None

                # Bookkeeping BEFORE announcement: record the metrics, then
                # wake the producer. set_result() can schedule the waiting
                # handler immediately, so anything we still need to do must
                # already be done.
                total_latency_ms = (req.t_completion - req.t_arrival) * 1000.0
                METRICS_WINDOW.append(CompletedRequest(total_latency_ms=total_latency_ms,tokens_generated=tokens_generated,t_completion=req.t_completion,))
                req.future.set_result(response)
        except Exception as e:
            # Batch-level failure policy: ZERO ORPHANS. If the shared call
            # blows up, every Future in the batch must be completed, or those
            # clients hang forever on an await that will never resolve. The
            # done() guard covers a partial failure — some Futures may already
            # be resolved from the loop above, and set_result twice raises.
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
        finally:
            # One task_done() per get(), success or failure — the queue's
            # internal counter must balance regardless of what happened.
            for _ in batch:
                REQUEST_QUEUE.task_done()


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
    app.state.worker_task = asyncio.create_task(worker(app.state.runner))
    logger.info("Background worker started.")

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
      - decode thread (yours, via run_in_executor): PRODUCES tokens into the sink
        with loop.call_soon_threadsafe(sink.put_nowait, token), then pushes a
        single None to mark end-of-stream.
      - THIS handler: CONSUMES the sink and frames each token onto the wire. It
        does no model work and no blocking work — hence `async def` (v1 was sync
        `def` because it drove the GPU inline; this one only awaits a Queue, so
        it belongs on the event loop where one thread serves many live streams).

    WIRE FORMAT (SSE) — this is the contract test_stream.py parses; the body you
    fill in below must emit exactly this:
      data: {"text": "<chunk>"}\n\n      <- one frame per token
      ...
      data: [DONE]\n\n                    <- terminal sentinel (sink yielded None)

    Each chunk is JSON-encoded (not raw text) so a token containing newlines or
    quotes can't corrupt the SSE framing — a raw newline would end the frame
    early. `json.dumps` is already imported at the top of this file.
    """
    # STEP 1 — CREATE THE SINK (your logic here)
    #   Make a fresh asyncio.Queue on the running loop; this is THIS request's
    #   private token channel. Grab the running loop too if the worker needs a
    #   handle for call_soon_threadsafe. (yours — this is the sink creation)

    # STEP 2 — ENQUEUE THROUGH THE WORKER PATH (your logic here)
    #   Build the in-flight request carrying the sink (instead of a Future),
    #   stamp t_arrival, and put it on REQUEST_QUEUE using the SAME backpressure
    #   policy /generate uses: put_nowait, and on asyncio.QueueFull raise
    #   HTTPException(status_code=503, ...). (yours — the enqueue logic)

    async def token_stream():
        """Async generator drained by StreamingResponse on the event loop.

        Its only job: pull tokens off the sink until the None sentinel, framing
        each as an SSE 'data:' line, then emit [DONE]. Keep at least one `yield`
        in here — without a `yield` Python makes this a coroutine, not an async
        generator, and StreamingResponse needs an async iterator.
        """
        # STEP 3 — DRAIN LOOP (your logic here)
        #   Loop: token = await sink.get(); a None means the producer is done ->
        #   break; otherwise yield  f"data: {json.dumps({'text': token})}\n\n".
        #   Consider wrapping this in try/finally so a client that disconnects
        #   mid-stream still unwinds cleanly (drain/cleanup is yours to decide).

        # STEP 4 — TERMINATE (your logic here)
        #   After the sentinel, emit the terminal frame:  yield "data: [DONE]\n\n"
        ...

    # --- StreamingResponse setup is wired for you; only the body above is yours ---
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

    Returns: p50_ms, p95_ms, p99_ms, sample_count, tokens_per_sec.
    Everything except sample_count can be None when the window is too thin —
    the caller must treat "not enough data yet" as a real state, not zero.
    """
    ...
    latencies: list[float] = []
    total_tokens = 0
    oldest_t_completion: float | None = None
    newest_t_completion: float | None = None

    # One walk, three quantities. The deque is in completion order, so the
    # first record seen is the oldest and the last one assigned is the newest.
    for record in METRICS_WINDOW:
        latencies.append(record.total_latency_ms)
        total_tokens += record.tokens_generated

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
        }

    latencies.sort()

    p50_ms = percentile(latencies, 0.50)
    p95_ms = percentile(latencies, 0.95)
    p99_ms = percentile(latencies, 0.99)

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
