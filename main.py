"""
main.py — FastAPI server for the mini inference engine.

v2 scope: every /generate request is serialized through ONE background worker
that owns the GPU. The flow is now producer/consumer:

    /generate handler (producer)
        --> builds a GenRequest (prompt + params + its own Future)
        --> puts it on REQUEST_QUEUE
        --> awaits the Future
                                  REQUEST_QUEUE (bounded asyncio.Queue)
    worker() (single consumer)
        --> gets the next GenRequest
        --> runs the model
        --> resolves that request's Future with the result

This is the foundation for dynamic batching next week: once all work funnels
through the one worker, the worker can start pulling N requests at once instead
of one. The request-queue/backpressure logic, the worker body, and the producer
body are YOURS (CLAUDE.md / Core List) — left as stubs below. The asyncio wiring
(queue object, startup/shutdown, create_task) is plumbing and is fully written.
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
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to continue.")
    max_tokens: int = Field(64, ge=1, le=2048, description="Max NEW tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float


# --------------------------- metrics: completed-request record (RESERVED) --------------------------- #
@dataclass(frozen=True)
class CompletedRequest:
    total_latency_ms: float
    tokens_generated: int
    t_completion: float


# class FakeRunner:
#     def generate_text(self, prompt, max_tokens, temperature):
#         time.sleep(0.05)  # pretend GPU work (blocking, like the real thing)
#         return f"fake reply to: {prompt[:20]}", 12


# --------------------------- the queue (plumbing) --------------------------- #
# Single hand-off point between the async HTTP handlers (producers) and the one
# background worker (consumer) that owns the GPU.
#
# Why BOUNDED, and why SMALL: maxsize caps how many requests can sit waiting in
# memory at once. When the queue is full, `await REQUEST_QUEUE.put(...)` blocks
# the producer until the worker drains one — i.e. the queue itself becomes the
# backpressure valve instead of letting an unbounded backlog grow until the box
# OOMs. 32 is a deliberately small placeholder: big enough to keep the worker fed,
# small enough that overload is felt quickly. Tune it once you have metrics.
#
# (Constructing the Queue at import time is fine on Python 3.10+: it binds to the
# running loop lazily on first use, not here.)
REQUEST_QUEUE: "asyncio.Queue[GenRequest]" = asyncio.Queue(maxsize=32)

# sliding window of completed requests; oldest fall off automatically
METRICS_WINDOW: "deque[CompletedRequest]" = deque(maxlen=1000)

# CHANGED 2026-07-02: was no batching knobs (worker consumed one request at a time);
# now the two dynamic-batching limits below; reason: worker's GRAB phase needs them.
MAX_BATCH_SIZE = 8  # max requests per forward pass
BATCH_WINDOW_S = 0.05  # accumulation deadline in seconds; clock starts at first arrival


# --------------------------- Core List: request container (STUB) --------------------------- #
@dataclass
class GenRequest:
    """
    One in-flight generation request as it travels through the queue.

    The producer (/generate) builds one of these, drops it on REQUEST_QUEUE, then
    awaits `.future`. The worker pulls it off, runs the model, and fulfils
    `.future` with the result — that round trip is the whole point of the queue.

    Fields (this is the container's "signature" — no behaviour to implement here):
      - prompt / max_tokens / temperature: the generation inputs.
      - future: the asyncio.Future the worker resolves with the GenerateResponse.
    """

    prompt: str
    max_tokens: int
    temperature: float
    # asyncio.Future = a one-shot "the result will arrive later" box. The producer
    # CREATES it and awaits it; the worker fills it via .set_result(...) (or
    # .set_exception(...) on failure), which is what wakes the awaiting producer.
    # It must be created on the running loop, so the producer makes it (see
    # /generate STEP 1) and passes it in here.
    future: "asyncio.Future[GenerateResponse]"

    # --- metrics timestamps (all optional, default None; NOBODY sets them here) ---
    # Each records one monotonic clock reading on this request's journey. They are
    # left as empty slots; the stamping code is Nony's to write at each stage.
    t_arrival: float | None = None       # stamped by ENDPOINT (/generate) at birth of GenRequest
    t_batch_entry: float | None = None   # stamped by WORKER when this request is pulled into a batch
    t_first_token: float | None = None   # stamped by RUNNER inside the decode loop (Stage 1a2, may defer)
    t_completion: float | None = None    # stamped by WORKER just before set_result


# --------------------------- Core List: the worker (STUB) --------------------------- #
async def worker(runner: ModelRunner) -> None:
    """
    The single background consumer. Owns the GPU: it is the ONLY thing that calls
    the model, so all requests are serialized through here. Runs forever until the
    shutdown hook cancels it.

    Input:
      - runner: the shared ModelRunner (loaded once at startup).
    Returns:
      - never returns normally; exits only via cancellation at an await point.

    # CHANGED 2026-07-02: was one-request-at-a-time (get -> run -> resolve) with an
    # unreachable raise NotImplementedError after the loop; now a two-phase batching
    # loop (STEP A grab / STEP B run) and the dead raise is deleted; reason: dynamic
    # batching — amortize each forward pass over up to MAX_BATCH_SIZE requests.

    Shape of each trip around the loop:
      STEP A — GRAB PHASE (yours): assemble a batch off REQUEST_QUEUE, bounded by
        MAX_BATCH_SIZE and a BATCH_WINDOW_S deadline that starts at first arrival.
      STEP B — RUN PHASE (written): execute the batch and resolve each request's
        Future. Temporarily sequential per request; later one batched forward pass.
    """
    while True:
        # ---------------- STEP A: GRAB PHASE (MY LOGIC — do not implement) ----------------
        # Assemble `batch: list[GenRequest]`:
        #   - Block on the empty queue; the FIRST arrival opens the batch and starts
        #     a FIXED deadline (BATCH_WINDOW_S measured from that first arrival —
        #     the deadline does NOT reset as more requests come in).
        #   - Keep grabbing further requests, each wait bounded by the REMAINING
        #     time until that deadline, until the batch is full (MAX_BATCH_SIZE)
        #     or the deadline expires.
        #   - TimeoutError is the go-signal ("window closed, ship what you have"),
        #     not an error.
        # Tools for this: asyncio.wait_for(...) + asyncio.TimeoutError for the
        # bounded waits; time.monotonic() for the fixed deadline / remaining-time math.
        # Move 1 — blocking first grab
        first_req = await REQUEST_QUEUE.get()
        first_req.t_batch_entry = time.monotonic()
        batch = [first_req]
        # Move 2 — start fixed clock
        deadline = time.monotonic() + BATCH_WINDOW_S
        # Move 3 — accumulation loop: keep going until batch is full
        while len(batch) < MAX_BATCH_SIZE:
            # Move 4 — bounded wait inside
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

            # Move 5 — timeout means window closed
            except asyncio.TimeoutError:
                break
            
        prompts = [req.prompt for req in batch]
        max_new_tokens = [req.max_tokens for req in batch]
        start = time.perf_counter()
        try:
            results = runner.generate_batch(prompts=prompts,max_new_tokens=max_new_tokens)
            model_latency_ms = (time.perf_counter() - start) * 1000.0
            assert len(results) == len(batch)

            for req, (text, tokens_generated) in zip(batch, results):
                response = GenerateResponse(text=text,tokens_generated=tokens_generated,latency_ms=model_latency_ms)
                req.t_completion = time.monotonic()
                assert req.t_arrival is not None

                total_latency_ms = (req.t_completion - req.t_arrival) * 1000.0
                METRICS_WINDOW.append(CompletedRequest(total_latency_ms=total_latency_ms,tokens_generated=tokens_generated,t_completion=req.t_completion,))
                req.future.set_result(response)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
        finally:
            for _ in batch:
                REQUEST_QUEUE.task_done()
            # ---------------- STEP B (Stage 4): BATCHED RUN — RESERVED, Nony's hands ----------------
        # INVARIANT: results[i] corresponds to batch[i] — generate_batch preserves prompt
        # order (verified: tokenizer + bins are positional, nothing sorts). If ANY
        # reordering is ever added (sort-by-length, continuous batching), positional
        # zip breaks SILENTLY — add ID plumbing first.
        #
        # RESERVED steps:
        #   1. Build parallel lists from batch: prompts, caps (req.max_tokens) — same order.
        #   2. Call the runner's batched path (design decision: which layer(s) to call
        #      to get BOTH texts and per-request token counts).
        #   3. assert len(results) == len(batch)  — count-in == count-out tripwire.
        #   4. Per request, IN ORDER: build GenerateResponse, stamp t_completion,
        #      build CompletedRequest + append to METRICS_WINDOW, then set_result.
        #      Bookkeeping before announcement, per request.
        #   5. Batch-level failure policy: if the batched call raises, EVERY Future in
        #      the batch must receive set_exception — zero orphans.
        #   6. task_done() once per request, exactly as before.
        # KNOWN LIMITATION (README): batched path is greedy — per-request temperature
        # accepted by API but not applied. Verify-first decision; sampling = future work.


# --------------------------- app lifecycle (plumbing) --------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    logger.info("Starting up — loading model ...")
    app.state.runner = ModelRunner()  # load weights once; reused for every request
    # app.state.runner = FakeRunner()
    # Launch the single background worker. create_task() SCHEDULES the worker()
    # coroutine to run concurrently on the event loop and returns IMMEDIATELY with a
    # Task handle — it does not block or run the worker inline here. We stash the
    # handle on app.state so the shutdown hook below can cancel it.
    app.state.worker_task = asyncio.create_task(worker(app.state.runner))
    logger.info("Background worker started.")

    yield  # <-- app serves requests for its whole lifetime here

    # --- shutdown ---
    logger.info("Shutting down — stopping worker ...")
    # cancel() requests cancellation: it arranges for a CancelledError to be raised
    # inside the worker at its next await point (typically `await REQUEST_QUEUE.get()`).
    app.state.worker_task.cancel()
    try:
        # Await the cancelled task so we actually wait for it to unwind before the
        # process exits. The CancelledError we just triggered propagates out of this
        # await — catching and ignoring it is the normal, clean way a cancelled task
        # is reaped (it is NOT an error here, it's the expected exit signal).
        await app.state.worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Worker stopped.")


app = FastAPI(title="Mini LLM Inference Engine", version="2.0", lifespan=lifespan)


# --------------------------- routes --------------------------- #
@app.get("/health")
def health(request: Request) -> dict:
    runner: ModelRunner = request.app.state.runner
    return {
        "status": "ok",
        "model_id": runner.model_id,
        "device": str(runner.device),
    }


# --------------------------- Core List: producer handler (STUB) --------------------------- #
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """
    Producer side of the queue. Now `async def` (not the old sync def): it runs ON
    the event loop so it can await the queue and the Future. It does NOT touch the
    model directly anymore — it hands work to the worker and waits for the answer.

    Input:
      - req: the validated GenerateRequest (prompt, max_tokens, temperature).
    Output:
      - the GenerateResponse the worker produced for this request.

    # STEP 1: create a fresh Future for THIS request (your logic here)
    #   - future = asyncio.get_running_loop().create_future()
    #   - a Future is the empty "result will arrive later" box; awaiting it parks
    #     THIS handler until the worker calls set_result/set_exception on it. Fresh
    #     one per request so results never cross wires between clients.

    # STEP 2: build the request object (your logic here)
    #   - wrap the prompt + params + that future in a GenRequest.

    # STEP 3: hand it to the worker by putting it on the queue (your logic here)
    #   - await REQUEST_QUEUE.put(gen_req)
    #   - this BLOCKS if the queue is full (the backpressure from maxsize). If you'd
    #     rather reject instead of wait, use REQUEST_QUEUE.put_nowait(...) inside
    #     try/except asyncio.QueueFull and raise HTTPException(503). Your call —
    #     this is the backpressure policy that's yours to design.

    # STEP 4: await the result and return it (your logic here)
    #   - result = await gen_req.future   <-- suspends here until the worker's
    #     STEP 4 resolves this exact Future; then control resumes with the value.
    #   - return result
    #   - (if you used set_exception in the worker, the await re-raises it here.)
    """
    future = asyncio.get_running_loop().create_future()
    # METRICS STEP 1: stamp t_arrival = time.monotonic() at birth of GenRequest
    gen_req = GenRequest(
        prompt = req.prompt,
        max_tokens = req.max_tokens,
        temperature = req.temperature,
        future = future,
        t_arrival=time.monotonic()
    )

    try:
        REQUEST_QUEUE.put_nowait(gen_req) #no await needed if you were doing .put() the you would have needed await 
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="server overloaded, try again later")
    response = await gen_req.future
    # METRICS STEP 4: compute latency_ms = (t_completion - t_arrival) * 1000
    # METRICS STEP 5: add tokens_generated + latency_ms to the response model/dict
    #   (GenerateResponse already declares both fields — see lines ~52-53 — so the
    #   plumbing is in place; this is just where you populate them. Your logic.)
    assert gen_req.t_completion is not None
    assert gen_req.t_arrival is not None
    response.latency_ms = (gen_req.t_completion - gen_req.t_arrival) * 1000
    return response


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest, request: Request) -> StreamingResponse:
    """Stream generated text token-by-token as Server-Sent Events (SSE).

    NOTE: streaming still calls the runner directly and does NOT go through the
    queue/worker yet — a single Future can't carry an incremental stream, that
    needs a per-request chunk channel. Left as-is for now; revisit once the
    blocking /generate path is flowing through the worker.

    Wire format (SSE):
      data: {"text": "<chunk>"}\n\n      <- one per token
      ...
      data: [DONE]\n\n                    <- terminal sentinel

    Each chunk is JSON-encoded (not raw text) so token text containing newlines
    or quotes can't corrupt the SSE framing. The client concatenates the "text"
    fields and stops on the [DONE] sentinel.

    Sync `def` on purpose: stream_tokens() does blocking GPU work, so Starlette
    iterates the returned sync generator in a threadpool and the event loop stays
    free.
    """
    runner: ModelRunner = request.app.state.runner

    # Prompt -> input_ids on the model's device (mirrors generate_text's tokenize
    # step; stream_tokens works in token-id space, just like generate_tokens).
    input_ids = runner.tokenizer(req.prompt, return_tensors="pt").input_ids.to(
        runner.device
    )

    def event_stream():
        # Re-emit each decoded token chunk as an SSE 'data:' frame.
        for chunk in runner.stream_tokens(
            input_ids=input_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        ):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        # Terminal sentinel so the client knows the stream is complete.
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush live
        },
    )


# --------------------------- Core List: metrics math (STUBS) --------------------------- #
def percentile(sorted_values: list[float], p: float) -> float | None:
    """RESERVED. Pure math, no globals. Assumes input already sorted ascending.
    Index convention: min(int(p * N), N - 1). Empty list -> None."""
    ...
    n = len(sorted_values)

    if n == 0:
        return None

    idx = min(int(p * n), n - 1)

    return sorted_values[idx]

def compute_metrics() -> dict:
    """RESERVED. Walks METRICS_WINDOW once: extract latencies, sort ONCE,
    call percentile() for p50/p95/p99; sum tokens_generated; span =
    newest t_completion - oldest t_completion; tokens_per_sec = None if
    fewer than 2 records; percentiles None if empty; always include
    sample_count. Returns dict: p50_ms, p95_ms, p99_ms, sample_count,
    tokens_per_sec."""
    ...
    latencies: list[float] = []
    total_tokens = 0
    oldest_t_completion: float | None = None
    newest_t_completion: float | None = None

    for record in METRICS_WINDOW:
        latencies.append(record.total_latency_ms)
        total_tokens += record.tokens_generated

        if oldest_t_completion is None:
            oldest_t_completion = record.t_completion

        newest_t_completion = record.t_completion

    sample_count = len(latencies)

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
    out = compute_metrics()
    out["queue_depth"] = REQUEST_QUEUE.qsize()
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
