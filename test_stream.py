"""
test_stream.py — manual verification that /generate/stream delivers tokens
INCREMENTALLY, not as one buffered blob at the end.

It hits POST /generate/stream, parses the SSE frames, and prints each chunk
with:
  - t+…      wall-clock time since the request was sent (so you can see WHEN
             each token landed)
  - +…ms     the gap since the PREVIOUS chunk (the real tell: if these are all
             ~0ms and clustered at the end, the server buffered; if they're
             spread out, tokens are streaming live)

At the end it prints TTFT (time to first token) and a simple verdict.

Usage:
    python main.py            # in one terminal (starts the server)
    python test_stream.py     # in another

    # options:
    python test_stream.py --prompt "Once upon a time" --max-tokens 40
    python test_stream.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream a single /generate/stream request and time each chunk.")
    p.add_argument("--url", default="http://localhost:8000", help="Server base URL.")
    p.add_argument("--prompt", default="The history of the internet begins", help="Prompt to stream.")
    p.add_argument("--max-tokens", type=int, default=40, help="Max new tokens to request.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    endpoint = args.url.rstrip("/") + "/generate/stream"
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    print(f"POST {endpoint}")
    print(f"     prompt={args.prompt!r}  max_tokens={args.max_tokens}  temperature={args.temperature}")
    print("-" * 72)

    # State for the timing report.
    t_start = time.perf_counter()
    t_prev = t_start           # timestamp of the previous chunk, for the delta
    t_first: float | None = None   # TTFT marker
    chunk_count = 0
    text_parts: list[str] = []

    # timeout=None so a slow generation isn't killed mid-stream. The read
    # timeout is what would otherwise fire between two slow tokens.
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", endpoint, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                print(f"HTTP {resp.status_code}: {body}")
                return

            # SSE frames are separated by blank lines; each data line is
            # "data: <payload>". aiter_lines() hands us one line at a time as
            # bytes arrive, which is exactly what lets us time each token.
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue  # blank separators / non-data fields — skip

                data = line[len("data:"):].strip()

                if data == "[DONE]":
                    break

                # Contract: each data frame is JSON {"text": "<chunk>"}.
                try:
                    chunk = json.loads(data)["text"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    print(f"  ?? unparseable frame: {data!r}")
                    continue

                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                delta_ms = (now - t_prev) * 1000.0
                since_start_ms = (now - t_start) * 1000.0
                t_prev = now
                chunk_count += 1
                text_parts.append(chunk)

                # repr() so whitespace/newlines in a token are visible rather
                # than reflowing the console output.
                print(f"  t+{since_start_ms:8.1f}ms  (+{delta_ms:7.1f}ms)  {chunk!r}")

    # ---- report ----
    print("-" * 72)
    total_ms = (time.perf_counter() - t_start) * 1000.0
    if chunk_count == 0:
        print("No chunks received — nothing to verify.")
        return

    ttft_ms = (t_first - t_start) * 1000.0  # type: ignore[operator]  (t_first set once chunk_count>0)
    print(f"chunks:      {chunk_count}")
    print(f"TTFT:        {ttft_ms:.1f}ms   (time to FIRST token)")
    print(f"total:       {total_ms:.1f}ms")
    print(f"full text:   {''.join(text_parts)!r}")

    # Heuristic verdict: if the whole thing arrived essentially at once, the
    # server buffered. If first-token latency is a real fraction of total, and
    # there are multiple chunks, it streamed.
    streamed = chunk_count > 1 and (total_ms - ttft_ms) > 5.0
    print(f"verdict:     {'STREAMED incrementally ✅' if streamed else 'looks BUFFERED / one-shot ⚠️'}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
