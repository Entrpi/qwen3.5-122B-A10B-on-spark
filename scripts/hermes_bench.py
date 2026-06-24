#!/usr/bin/env python3
"""Real-world DFlash/MTP acceptance bench on the actual Hermes agent workload.

Reconstructs real conversation contexts from ~/.hermes/state.db and has the loaded
server regenerate the *next assistant turn* (so the model that produced the history is
irrelevant — only the realistic context matters). Measures mean acceptance length
(authoritative vLLM /metrics deltas) + decode tok/s, aggregated over N real turns.
Privacy: runs entirely on the box; prints only metrics, not conversation content.
"""
import argparse, json, sqlite3, sys, time, urllib.request, urllib.error

DB = "/home/ent/.hermes/state.db"


def scrape(base):
    out = {"acc": 0.0, "drafts": 0.0, "dtoks": 0.0}
    try:
        txt = urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10).read().decode("utf-8", "replace")
    except Exception:
        return out
    for ln in txt.splitlines():
        if ln.startswith("#"):
            continue
        v = ln.split()[-1] if ln.split() else "0"
        try:
            val = float(v)
        except ValueError:
            continue
        if "spec_decode_num_accepted_tokens_total" in ln:
            out["acc"] += val
        elif "spec_decode_num_drafts_total" in ln:
            out["drafts"] += val
        elif "spec_decode_num_draft_tokens_total" in ln:
            out["dtoks"] += val
    return out


def build_messages(cur, session_id, system_prompt, char_budget=24000):
    rows = list(cur.execute(
        "SELECT role, content, tool_calls, tool_call_id, tool_name FROM messages "
        "WHERE session_id=? ORDER BY id", (session_id,)))
    # find the LAST assistant turn -> generate it; prompt = everything before it
    last_asst = None
    for i, r in enumerate(rows):
        if r[0] == "assistant":
            last_asst = i
    if last_asst is None or last_asst == 0:
        return None
    pre = rows[:last_asst]
    msgs = []
    for role, content, tool_calls, tool_call_id, tool_name in pre:
        content = content or ""
        if role == "user":
            msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            m = {"role": "assistant", "content": content}
            if tool_calls:
                try:
                    tc = json.loads(tool_calls)
                    if isinstance(tc, list) and tc:
                        m["tool_calls"] = tc
                        if not content:
                            m["content"] = ""
                except Exception:
                    pass
            msgs.append(m)
        elif role == "tool":
            msgs.append({"role": "tool", "content": content,
                         "tool_call_id": tool_call_id or "call_0"})
        # skip session_meta
    if not msgs:
        return None
    # truncate oldest non-system messages to fit budget
    sys_msg = [{"role": "system", "content": system_prompt}] if system_prompt else []
    while sum(len(json.dumps(m)) for m in msgs) > char_budget and len(msgs) > 1:
        msgs.pop(0)
    return sys_msg + msgs


def stream_chat(base, model, messages, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--min-msgs", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--label", default="HERMES real")
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    sess = list(cur.execute(
        "SELECT s.id, s.system_prompt FROM sessions s "
        "WHERE s.message_count >= ? ORDER BY s.started_at DESC LIMIT 40", (args.min_msgs,)))

    m0 = scrape(args.base_url)
    decode_tps, used = [], 0
    for sid, sysp in sess:
        if used >= args.n_samples:
            break
        try:
            msgs = build_messages(cur, sid, sysp)
        except Exception:
            continue
        if not msgs:
            continue
        try:
            toks, tf, tl = stream_chat(args.base_url, args.model, msgs, args.max_tokens, args.timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"  skip {sid[:24]}: {type(e).__name__}", file=sys.stderr)
            continue
        if toks and tf and tl and tl > tf:
            dec = (toks - 1) / (tl - tf)
            decode_tps.append(dec)
            used += 1
            print(f"  sample {used}: {toks} tok  decode={dec:5.1f} tok/s  ({sid[:28]})")
    m1 = scrape(args.base_url)

    dacc = m1["acc"] - m0["acc"]
    ddr = m1["drafts"] - m0["drafts"]
    ddt = m1["dtoks"] - m0["dtoks"]
    import statistics
    print(f"\n=== {args.label} (n={used} real turns) ===")
    if decode_tps:
        print(f"decode tok/s : median {statistics.median(decode_tps):.1f}  mean {statistics.mean(decode_tps):.1f}  "
              f"min {min(decode_tps):.1f}  max {max(decode_tps):.1f}")
    if ddr > 0:
        print(f"accept length: {1 + dacc/ddr:.2f}  (accepted {dacc:.0f} / drafts {ddr:.0f}; "
              f"draft tokens {ddt:.0f}; accept rate {dacc/ddt:.1%})")
    else:
        print("accept length: (no draft activity captured)")


if __name__ == "__main__":
    main()
