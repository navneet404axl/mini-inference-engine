import asyncio
import httpx

URL = "http://localhost:8000/generate"
N = 30  # number of simultaneous requests

async def one(client, i):
    try:
        r = await client.post(URL, json={
            "prompt": f"Request {i}",
            "max_tokens": 20,
            "temperature": 0.7,
        })
        return i, r.status_code
    except Exception as e:
        return i, f"ERR {type(e).__name__}"

async def main():
    # no timeout so the slow 200s don't get killed before they return
    async with httpx.AsyncClient(timeout=None) as client:
        # fire all N at once — this is the key difference from the bash loop
        results = await asyncio.gather(*(one(client, i) for i in range(1, N + 1)))

    codes = [c for _, c in results]
    n_200 = codes.count(200)
    n_503 = codes.count(503)
    print(f"\n200s: {n_200}   503s: {n_503}   other: {len(codes) - n_200 - n_503}")
    for i, c in sorted(results):
        print(f"req {i}: {c}")

asyncio.run(main())