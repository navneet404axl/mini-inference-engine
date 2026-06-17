# Mini LLM Inference Engine — MVP Spec

## What it does
A backend server that receives a text prompt over HTTP, runs it through
Qwen2.5-1.5B on a GPU, and returns the generated text. Built to serve
many requests at once and to measure its own speed.

## API (v1)
POST /generate
  Request:  { "prompt": str, "max_tokens": int, "temperature": float }
  Response: { "text": str, "tokens_generated": int, "latency_ms": float }
GET /metrics
  Returns: tokens/sec, p50/p95/p99 latency, queue depth

## Components
1. FastAPI server — receives requests
2. Request queue — holds requests when the model is busy
3. Model runner — loads Qwen2.5-1.5B, runs the generation loop
4. (Week 2) Streamer — sends tokens as they're generated
5. (Week 3) Batcher — groups concurrent requests into one forward pass
6. Metrics — records timings at each stage

## In scope for v1
Single model, single GPU, streaming, dynamic batching, metrics endpoint

## Out of scope (deliberately)
- No frontend / web UI (tested via scripts, not a browser)
- No authentication or user accounts
- No multiple models (just Qwen2.5-1.5B)
- No multi-GPU
- No quantization

## Success criteria
- Handles ~20 concurrent requests correctly
- Batched throughput measurably higher than sequential
- Reports p50 / p95 / p99 latency under load

## Open questions (answer as I build)
- Batch on a fixed timer, or when N requests accumulate?
- What happens when the queue is full — reject, or make the user wait?
- How many tokens/sec can a T4 realistically push for this model?