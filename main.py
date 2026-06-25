"""
main.py — FastAPI server for the mini inference engine.

v1 scope (per README): single model, single GPU, POST /generate.
The request queue, batching, and the /metrics aggregation (p50/p95/p99) are
deliberately NOT here yet — those are yours to build (CLAUDE.md). This file is
just the HTTP scaffolding wired to the ModelRunner.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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


# --------------------------- app lifecycle --------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup and hang it off app.state for every handler.
    logger.info("Starting up — loading model ...")
    app.state.runner = ModelRunner()
    yield
    # Nothing to tear down explicitly; let the process drop the GPU memory on exit.
    logger.info("Shutting down.")


app = FastAPI(title="Mini LLM Inference Engine", version="1.0", lifespan=lifespan)


# --------------------------- routes --------------------------- #
@app.get("/health")
def health(request: Request) -> dict:
    runner: ModelRunner = request.app.state.runner
    return {
        "status": "ok",
        "model_id": runner.model_id,
        "device": str(runner.device),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """Generate text for a single prompt.

    Defined as a sync `def` on purpose: FastAPI runs sync handlers in a
    threadpool, so a blocking generation call won't freeze the event loop. When
    you add the request queue / batcher, this is where you'll route through it
    instead of calling the runner directly.
    """
    runner: ModelRunner = request.app.state.runner

    # Single-request timing only — this is the response contract's latency_ms,
    # not the aggregate p50/p95/p99 math (that's yours to build for /metrics).
    start = time.perf_counter()
    text, tokens_generated = runner.generate_text(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    return GenerateResponse(
        text=text,
        tokens_generated=tokens_generated,
        latency_ms=latency_ms,
    )


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest, request: Request) -> StreamingResponse:
    """Stream generated text token-by-token as Server-Sent Events (SSE).

    Same inputs as /generate, but incremental: instead of one JSON body at the
    end, the client gets one SSE event per token as it is produced.

    Wire format (SSE):
      data: {"text": "<chunk>"}\n\n      <- one per token
      ...
      data: [DONE]\n\n                    <- terminal sentinel

    Each chunk is JSON-encoded (not raw text) so token text containing newlines
    or quotes can't corrupt the SSE framing. The client concatenates the "text"
    fields and stops on the [DONE] sentinel.

    Sync `def` on purpose: stream_tokens() does blocking GPU work, so Starlette
    iterates the returned sync generator in a threadpool and the event loop stays
    free. When you add the queue/batcher, route the per-token stream through it.
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
