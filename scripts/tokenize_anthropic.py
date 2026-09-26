"""Count tokens for every unique solution with Anthropic's count_tokens endpoint.

Resumable: appends to build/counts/anthropic-<model>.jsonl and skips shas already present.
count_tokens is free but rate limited per tier.

    python scripts/tokenize_anthropic.py
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "RosettaCodeData"
COUNTS = ROOT / "build" / "counts"


async def raw_count(client: anthropic.AsyncAnthropic, model: str, text: str) -> int:
    resp = await client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    )
    return resp.input_tokens


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--rpm", type=int, default=7800, help="client-side pacing below the org's count_tokens limit")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new files (0 = all)")
    args = ap.parse_args()

    client = anthropic.AsyncAnthropic(base_url="https://api.anthropic.com", max_retries=3)

    # A single ASCII character is one token in every BPE vocabulary, so raw - 1 is the message framing.
    probes = [await raw_count(client, args.model, c) for c in "xa1;"]
    if len(set(probes)) != 1:
        sys.exit(f"inconsistent framing probes {probes}")
    overhead = probes[0] - 1

    COUNTS.mkdir(parents=True, exist_ok=True)
    tok_id = f"anthropic-{args.model}"
    out_path = COUNTS / f"{tok_id}.jsonl"
    (COUNTS / f"{tok_id}.meta.json").write_text(json.dumps({"model": args.model, "overhead": overhead}))

    done = set()
    if out_path.exists():
        # An interrupted run can leave a partial last line; keep only complete rows.
        rows = []
        for line in open(out_path):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        done = {r["sha"] for r in rows}
    todo = {}
    for line in open(ROOT / "build" / "files.jsonl"):
        r = json.loads(line)
        if r["sha"] not in done:
            todo.setdefault(r["sha"], r["path"])
    items = list(todo.items())
    if args.limit:
        items = items[: args.limit]
    print(f"{tok_id}: overhead={overhead}, done={len(done)}, todo={len(items)}", file=sys.stderr)

    queue: asyncio.Queue = asyncio.Queue()
    for it in items:
        queue.put_nowait(it)
    out = open(out_path, "a")
    stats = {"ok": 0, "err": 0}
    t0 = time.time()
    interval = 60 / args.rpm
    next_slot = [time.monotonic()]

    async def pace():
        now = time.monotonic()
        slot = max(now, next_slot[0])
        next_slot[0] = slot + interval
        if slot > now:
            await asyncio.sleep(slot - now)

    async def worker():
        while True:
            try:
                sha, path = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            text = (DATA / path).read_bytes().decode("utf-8", errors="replace")
            try:
                await pace()
                n = await raw_count(client, args.model, text) - overhead
            except anthropic.RateLimitError:
                await asyncio.sleep(5)
                queue.put_nowait((sha, path))
                continue
            except anthropic.BadRequestError as e:
                stats["err"] += 1
                print(f"skip {path}: {e.message}", file=sys.stderr)
                continue
            out.write(json.dumps({"sha": sha, "n": n}) + "\n")
            stats["ok"] += 1
            if stats["ok"] % 2000 == 0:
                out.flush()
                rate = stats["ok"] / (time.time() - t0) * 60
                print(f"{stats['ok']}/{len(items)} ({rate:.0f}/min)", file=sys.stderr)

    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    out.close()
    print(f"finished ok={stats['ok']} err={stats['err']}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
