#!/usr/bin/env python3
"""Inspect the Intel INT4 AutoRound checkpoint to decide if albond's hybrid
INT4+FP8 build will work: it swaps BF16 *dense* (non-expert) tensors for FP8 by
name. If AutoRound quantized the dense linears to INT4 (.qweight), the swap is a
near no-op. We need attention/shared_expert dense weights stored as BF16 (.weight).

Also pulls the FP8 repo's index to confirm it exists and that names line up.
"""
import json
import sys
from collections import Counter
from pathlib import Path

INT4_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/ent/.cache/huggingface/hub/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/"
    "snapshots/3045d02bb737effc4581da91bddbad3be02934e4")
FP8_REPO = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-122B-A10B-FP8"

idx = json.loads((INT4_DIR / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
names = list(wm.keys())

# Non-expert = no ".experts." (routed experts stay INT4 in the hybrid)
nonexp = [n for n in names if ".experts." not in n]

def bucket(n):
    if ".experts." in n:
        return "ROUTED-EXPERT"
    if "shared_expert" in n:
        return "shared_expert"
    if "self_attn" in n or "linear_attn" in n or ".attn" in n:
        return "attention"
    if "embed_tokens" in n or "lm_head" in n:
        return "embed/head"
    if "mtp" in n:
        return "mtp"
    return "other"

print(f"=== Intel INT4 checkpoint: {INT4_DIR.name} ===")
print(f"total tensors: {len(names)}  non-expert: {len(nonexp)}")

# Suffix histogram tells us quant scheme: .qweight/.scales/.qzeros => INT4; .weight => dense
suffix = Counter(n.rsplit(".", 1)[-1] for n in nonexp)
print("\nnon-expert tensor SUFFIX histogram (qweight/scales/qzeros = INT4-packed; weight = dense):")
for s, c in suffix.most_common():
    print(f"  .{s:20s} {c}")

# For each functional group, does it have .weight (BF16 dense) or .qweight (INT4)?
print("\nper-group quant scheme (sample names):")
groups = {}
for n in nonexp:
    g = bucket(n)
    groups.setdefault(g, {"weight": 0, "qweight": 0, "scale": 0, "other": 0, "ex": None})
    suf = n.rsplit(".", 1)[-1]
    if suf == "weight":
        groups[g]["weight"] += 1
    elif suf == "qweight":
        groups[g]["qweight"] += 1
    elif "scale" in suf or suf in ("qzeros",):
        groups[g]["scale"] += 1
    else:
        groups[g]["other"] += 1
    if groups[g]["ex"] is None and suf in ("weight", "qweight"):
        groups[g]["ex"] = n
for g, d in sorted(groups.items()):
    scheme = "INT4(.qweight)" if d["qweight"] else ("DENSE(.weight)" if d["weight"] else "?")
    print(f"  {g:16s} weight={d['weight']:4d} qweight={d['qweight']:4d} scale={d['scale']:4d}  -> {scheme}")
    print(f"      e.g. {d['ex']}")

# Dtypes of a few non-expert .weight tensors (open the shard header only).
print("\ndtypes of sample non-expert '.weight' tensors (FP8 swap needs BF16 here):")
from safetensors import safe_open  # noqa: E402
sample = [n for n in nonexp if n.endswith(".weight")
          and ("self_attn" in n or "shared_expert" in n or "embed" in n or "lm_head" in n)]
seen_shards = {}
shown = 0
for n in sample:
    shard = wm[n]
    f = seen_shards.get(shard)
    if f is None:
        f = safe_open(str(INT4_DIR / shard), framework="pt")
        seen_shards[shard] = f
    try:
        t = f.get_slice(n)
        print(f"  {n:60s} {t.get_dtype()} {tuple(t.get_shape())}")
    except Exception as e:
        print(f"  {n:60s} <err {e}>")
    shown += 1
    if shown >= 12:
        break

# FP8 repo: confirm exists + list its non-expert names/dtypes for name-match sanity.
print(f"\n=== FP8 repo manifest: {FP8_REPO} ===")
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(FP8_REPO, "model.safetensors.index.json")
    fidx = json.loads(Path(p).read_text())
    fwm = fidx["weight_map"]
    fnon = [n for n in fwm if ".experts." not in n]
    print(f"FP8 total tensors: {len(fwm)}  non-expert: {len(fnon)}")
    fsuf = Counter(n.rsplit('.', 1)[-1] for n in fnon)
    print("FP8 non-expert suffix histogram:")
    for s, c in fsuf.most_common(10):
        print(f"  .{s:20s} {c}")
    # how many FP8 non-expert .weight names also exist in INT4 as .weight?
    int4_weight = {n for n in nonexp if n.endswith('.weight')}
    fp8_weight = {n for n in fnon if n.endswith('.weight')}
    match = int4_weight & fp8_weight
    print(f"\nname overlap (FP8 '.weight' that also exist as '.weight' in INT4): {len(match)} / {len(fp8_weight)} FP8 weights")
    only_fp8 = sorted(fp8_weight - int4_weight)[:8]
    print(f"FP8 '.weight' NOT present as '.weight' in INT4 (would not swap): {len(fp8_weight - int4_weight)}")
    for n in only_fp8:
        print(f"    {n}  (INT4 has: {'qweight' if n[:-7]+'.qweight' in set(names) else 'MISSING'})")
except Exception as e:
    print(f"<FP8 repo fetch failed: {e}>")
