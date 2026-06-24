#!/usr/bin/env python3
"""Faithful reproduction of albond's bench_qwen35.sh methodology so our DFlash/MTP
numbers are directly comparable to his reported 51.58 tok/s.

His method (verbatim): non-streaming /v1/chat/completions, time the WHOLE request,
tok/s = completion_tokens / wall_time (INCLUDES prefill + overhead = END-TO-END).
5 prompts (Q&A 256 / Code 512 / JSON 1024 / Math 64 / LongCode 2048), temp 0, run 1
discarded as JIT warmup. We add: decode-only tok/s isn't measured here on purpose
(his isn't either) + an aggregate spec-accept-len from /metrics deltas for context.
"""
import json, sys, time, urllib.request, statistics

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "qwen"
PROMPTS = [
    ("Q&A",      "What are the main differences between TCP and UDP? Be concise.", 256),
    ("Code",     "Write a Python function that implements binary search on a sorted list. Include type hints and docstring.", 512),
    ("JSON",     "Generate a JSON array of 10 fictional employees with fields: name, age, department, salary, email, skills (array of 3). Output ONLY valid JSON, no explanation.", 1024),
    ("Math",     "What is 7823 * 4519? Show only the answer.", 64),
    ("LongCode", "Write a complete Python implementation of a red-black tree with insert, delete, search, and in-order traversal. Include all rotation methods.", 2048),
]


def scrape():
    acc = dr = 0.0
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
        for ln in txt.splitlines():
            if ln.startswith("#") or not ln.split():
                continue
            v = float(ln.split()[-1])
            if "spec_decode_num_accepted_tokens_total" in ln:
                acc += v
            elif "spec_decode_num_drafts_total" in ln:
                dr += v
    except Exception:
        pass
    return acc, dr


def chat_e2e(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    elapsed = time.perf_counter() - t0
    ct = r["usage"]["completion_tokens"]
    return ct, elapsed, ct / elapsed if elapsed > 0 else 0.0


def main():
    label = sys.argv[2] if len(sys.argv) > 2 else "server"
    print(f"=== albond-method e2e bench :: {label} ===")
    a0, d0 = scrape()
    run2 = {}
    for run in (1, 2):
        tag = "WARMUP(discard)" if run == 1 else "RUN2"
        for name, prompt, mt in PROMPTS:
            try:
                ct, el, tps = chat_e2e(prompt, mt)
            except Exception as e:
                print(f"  [{name}] FAILED: {type(e).__name__}: {e}"); continue
            if run == 2:
                run2[name] = tps
            print(f"  {tag:16s} [{name:8s}] {ct:4d} tok in {el:6.2f}s = {tps:6.1f} tok/s (e2e)")
    a1, d1 = scrape()
    print(f"\n{label} RUN2 e2e tok/s: " + "  ".join(f"{k}={v:.1f}" for k, v in run2.items()))
    if run2:
        print(f"  cross-prompt mean (RUN2) = {statistics.mean(run2.values()):.1f} tok/s  (albond reports 51.58)")
    if d1 - d0 > 0:
        print(f"  aggregate spec accept length over bench = {1 + (a1-a0)/(d1-d0):.2f}")


if __name__ == "__main__":
    main()
