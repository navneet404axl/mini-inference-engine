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

    # STEP 1: loop forever  ->  `while True:`

    # STEP 2: get the next request off the queue (your logic here)
    #   - req = await REQUEST_QUEUE.get()
    #   - the `await` parks this coroutine until a request is available — no
    #     busy-waiting, the event loop runs other things meanwhile.

    # STEP 3: run the model on this request (your logic here)
    #   - call runner.generate_text(prompt=..., max_tokens=..., temperature=...)
    #   - time it (time.perf_counter()) to fill latency_ms, like the old handler did
    #   - NOTE: generate_text() is BLOCKING GPU work. Awaiting nothing inside it
    #     means it blocks the event loop for its whole duration. That's acceptable
    #     for now (one worker, fully serialized) but worth knowing — later you may
    #     push it onto a thread via loop.run_in_executor(...) so the loop can keep
    #     accepting requests. Your design call.

    # STEP 4: resolve THIS request's Future with the result (your logic here)
    #   - req.future.set_result(GenerateResponse(...))  <-- this wakes the producer
    #     that's awaiting req.future over in /generate.
    #   - wrap STEP 3 in try/except and on error call req.future.set_exception(err)
    #     instead — otherwise one bad request hangs that client's await forever AND
    #     an unhandled exception here would kill the worker for everyone.

    # STEP 5: mark the queue item done (your logic here)
    #   - REQUEST_QUEUE.task_done()  (pairs with the get() in STEP 2)
    """
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop()
    # loop.run_forever()
    # REQUEST_QUEUE.get()
    while True:
        req = await REQUEST_QUEUE.get()
        start = time.perf_counter()
        try:
            text,tokens_generated = runner.generate_text(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                )
            latency_ms = (time.perf_counter() - start) * 1000.0
            response = GenerateResponse(
                text=text,
                tokens_generated=tokens_generated,
                latency_ms=latency_ms,
                )
            req.future.set_result(response)
        except Exception as e:
            req.future.set_exception(e)

        finally:
            REQUEST_QUEUE.task_done()
    raise NotImplementedError(
        "worker is yours to implement — see the STEP comments above."
    )


# --------------------------- app lifecycle (plumbing) --------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    logger.info("Starting up — loading model ...")
    app.state.runner = ModelRunner()  # load weights once; reused for every request

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
    gen_req = GenRequest(
        prompt = req.prompt,
        max_tokens = req.max_tokens,
        temperature = req.temperature,
        future = future
    )
    try:
        REQUEST_QUEUE.put_nowait(gen_req) #no await needed if you were doing .put() the you would have needed await 
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="server overloaded, try again later")
    returned_future = await gen_req.future
    return returned_future

    raise NotImplementedError(
        "generate (producer) is yours to implement — see the STEP comments above."
    )


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
