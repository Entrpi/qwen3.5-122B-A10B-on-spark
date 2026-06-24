#!/usr/bin/env python3
"""Single-stream (c=1) decode-rate benchmark for an OpenAI-compatible vLLM server.

Measures *decode* tok/s (excludes TTFT/prefill) over N sequential requests, and
optionally scrapes vLLM /metrics for speculative-decode acceptance length.

stdlib only — runs anywhere with python3, no pip installs.

  python3 bench_decode.py --base-url http://127.0.0.1:8000 \
      --model Intel/Qwen3.5-122B-A10B-int4-AutoRound \
      --max-tokens 256 --runs 5 --label "int4 baseline"
"""
import argparse, json, statistics, sys, time, urllib.request, urllib.error

PROMPT = ("You are a careful writer. Write a detailed, continuous explanation of how "
          "a modern mixture-of-experts transformer performs autoregressive decoding, "
          "covering routing, KV cache, and memory bandwidth. Begin now:\n\n")


def post_stream(base_url, model, prompt, max_tokens, timeout):
    """Stream /v1/completions; return (completion_tokens, t_first, t_last)."""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        # force the full token budget so we measure steady-state decode
        "ignore_eos": True, "min_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions",
                                 data=body, headers={"Content-Type": "application/json",
                                                     "Authorization": "Bearer x"})
    t_first = t_last = None
    completion_tokens = 0
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                chunks += 1
    if completion_tokens == 0:
        completion_tokens = chunks  # fallback: 1 chunk ~= 1 token
    return completion_tokens, t_first, t_last


def scrape_metrics(base_url, timeout=10):
    """Return dict of spec-decode counters from vLLM /metrics, if present."""
    keys = ("num_accepted_tokens", "num_draft_tokens", "num_drafts",
            "accepted_tokens", "draft_tokens")
    out = {}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith("#"):
                    continue
                if "spec_decode" in line or "speculat" in line:
                    name, _, val = line.partition(" ")
                    try:
                        out[name] = out.get(name, 0.0) + float(val)
                    except ValueError:
                        pass
    except (urllib.error.URLError, OSError):
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--label", default="")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    m_before = scrape_metrics(args.base_url)
    for _ in range(args.warmup):
        try:
            post_stream(args.base_url, args.model, args.prompt, 32, args.timeout)
        except Exception as e:
            print(f"warmup failed: {e}", file=sys.stderr); sys.exit(2)

    decode_tps, ttfts, e2e_tps = [], [], []
    for i in range(args.runs):
        t0 = time.perf_counter()
        toks, tf, tl = post_stream(args.base_url, args.model, args.prompt, args.max_tokens, args.timeout)
        t1 = time.perf_counter()
        if not toks or tf is None or tl is None or tl <= tf:
            print(f"  run {i}: degenerate (toks={toks})", file=sys.stderr); continue
        dec = (toks - 1) / (tl - tf)
        decode_tps.append(dec); ttfts.append((tf - t0) * 1000); e2e_tps.append(toks / (t1 - t0))
        print(f"  run {i}: {toks} tok  decode={dec:6.1f} tok/s  ttft={ (tf-t0)*1000:6.0f} ms")
    m_after = scrape_metrics(args.base_url)

    if not decode_tps:
        print("no successful runs", file=sys.stderr); sys.exit(1)
    print(f"\n=== {args.label or args.model} ===")
    print(f"decode tok/s : median {statistics.median(decode_tps):.1f}  "
          f"mean {statistics.mean(decode_tps):.1f}  min {min(decode_tps):.1f}  max {max(decode_tps):.1f}")
    print(f"ttft ms      : median {statistics.median(ttfts):.0f}")
    print(f"e2e tok/s    : median {statistics.median(e2e_tps):.1f}")

    # spec-decode acceptance length, if counters moved
    def delta(k):
        return m_after.get(k, 0.0) - m_before.get(k, 0.0)
    acc = next((delta(k) for k in m_after if "accepted" in k), 0.0)
    drafts = next((delta(k) for k in m_after if "num_drafts" in k or ("draft" in k and "tokens" not in k)), 0.0)
    dtoks = next((delta(k) for k in m_after if "draft_tokens" in k or "num_draft_tokens" in k), 0.0)
    if acc or dtoks:
        al = (acc / drafts) if drafts else float("nan")
        rate = (acc / dtoks) if dtoks else float("nan")
        print(f"spec accept  : +{acc:.0f} accepted, +{dtoks:.0f} drafted, "
              f"mean accept len ~{al:.2f}, accept rate ~{rate:.1%}")
    else:
        print("spec accept  : (no spec-decode counters — baseline/no drafter)")


if __name__ == "__main__":
    main()
