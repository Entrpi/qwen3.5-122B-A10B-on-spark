#!/usr/bin/env python3
"""Per-workload concurrency sweep. Reports DECODE-only tok/s (streaming, excludes
prefill/TTFT) for prose / code / agentic at K concurrent streams. "agentic" uses
real tool-call contexts from ~/.hermes/state.db (the workload DFlash wins biggest
on); prose/code use fixed prompts. Per-stream = mean decode rate of the K streams;
aggregate = sum (concurrent total throughput).

  python3 conc_workloads.py --base-url http://127.0.0.1:8000 --levels 1,2,3
"""
import argparse
import concurrent.futures as cf
import json
import sqlite3
import statistics
import sys
import time
import urllib.request

DB = "/home/ent/.hermes/state.db"

PROSE = [{"role": "user", "content":
          "Write a detailed, flowing essay about the history and cultural "
          "significance of tea across civilizations."}]
CODE = [{"role": "user", "content":
         "Implement a complete red-black tree in Python: insert, delete, search, "
         "rotations, with type hints and docstrings. Output only the code."}]


def stream_chat(base, model, messages, max_tokens, timeout=600):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_first = t_last = None
    toks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:"):
                continue
            d = ln[5:].strip()
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except json.JSONDecodeError:
                continue
            u = o.get("usage")
            if u and u.get("completion_tokens"):
                toks = u["completion_tokens"]
            ch = o.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
    return toks, t_first, t_last


def decode_rate(res):
    toks, tf, tl = res
    return (toks - 1) / (tl - tf) if (toks and tf and tl and tl > tf) else 0.0


def load_agentic(min_msgs=8, char_budget=60000):
    """Pick a representative real tool-call context (truncate to ~budget)."""
    db = sqlite3.connect(DB)
    cur = db.cursor()
    for sid, sysp in cur.execute(
            "SELECT id, system_prompt FROM sessions WHERE message_count >= ? "
            "ORDER BY started_at DESC LIMIT 40", (min_msgs,)):
        rows = list(cur.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages "
            "WHERE session_id=? ORDER BY id", (sid,)))
        last = max((i for i, r in enumerate(rows) if r[0] == "assistant"), default=None)
        if not last:
            continue
        msgs = []
        for role, content, tc, tcid in rows[:last]:
            content = content or ""
            if role == "user":
                msgs.append({"role": "user", "content": content})
            elif role == "assistant":
                m = {"role": "assistant", "content": content}
                if tc:
                    try:
                        j = json.loads(tc)
                        if isinstance(j, list) and j:
                            m["tool_calls"] = j
                    except Exception:
                        pass
                msgs.append(m)
            elif role == "tool":
                msgs.append({"role": "tool", "content": content, "tool_call_id": tcid or "call_0"})
        if not msgs:
            continue
        sysm = [{"role": "system", "content": sysp}] if sysp else []
        while sum(len(json.dumps(m)) for m in msgs) > char_budget and len(msgs) > 1:
            msgs.pop(0)
        approx = sum(len(json.dumps(m)) for m in (sysm + msgs)) // 4
        return sysm + msgs, approx
    return None, 0


def sweep(base, model, label, messages, max_tokens, levels):
    stream_chat(base, model, messages, 16)  # warm
    row = [label]
    for K in levels:
        with cf.ThreadPoolExecutor(max_workers=K) as ex:
            res = list(ex.map(lambda _: stream_chat(base, model, messages, max_tokens), range(K)))
        rates = [decode_rate(r) for r in res]
        row.append((statistics.mean(rates), sum(rates)))
        print(f"    {label:8s} K={K}: per-stream {statistics.mean(rates):5.1f} tok/s "
              f"(min {min(rates):.1f}/max {max(rates):.1f})  aggregate {sum(rates):6.1f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--levels", default="1,2,3")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    agentic, approx = load_agentic()
    print(f"=== concurrency sweep (decode-only tok/s, streaming) levels={levels} ===")
    if agentic:
        print(f"  (agentic context ~{approx} tokens, real tool-call turn)")
    sweep(args.base_url, args.model, "prose", PROSE, args.max_tokens, levels)
    sweep(args.base_url, args.model, "code", CODE, args.max_tokens, levels)
    if agentic:
        sweep(args.base_url, args.model, "agentic", agentic, args.max_tokens, levels)
    else:
        print("  (no Hermes DB found — skipped agentic)", file=sys.stderr)


if __name__ == "__main__":
    main()
