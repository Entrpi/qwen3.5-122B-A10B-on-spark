# FINDINGS — DFlash + dense levers for Qwen3.5-122B-A10B on DGX Spark

Single-stream (c=1) decode of `Qwen3.5-122B-A10B` (hybrid GDN + mamba + 128-expert
MoE, ~10B active) on GB10 / SM121, 128 GiB unified, ~273 GB/s. The agent this
backs (Hermes) is ~73 % tool-calls. All numbers temperature 0.

> **Credit.** This builds on [albond's recipe](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)
> (hybrid INT4+FP8 + INT8 lm-head + MTP-2 on vLLM 0.19, and the e2e benchmark
> method). The new work here is forward-porting the dense levers to vLLM 0.23,
> swapping MTP for DFlash, and composing them — landing at **59.0 e2e tok/s vs
> albond's 51.58** on his own harness (+14%), and ~2× on real agent traffic.
> See README → *Building on the albond recipe* for the side-by-side.

## 1. Getting DFlash to run on the hybrid 122B in vLLM

The DFlash drafter is **non-causal** (it block-drafts 16 tokens in one parallel
forward) — only the `FLASH_ATTN` (FA2) backend supports non-causal attention.
But the hybrid GDN+mamba+MoE target's KV-cache page geometry won't *unify* with
the drafter's attention spec:

- vLLM auto-aligns the hybrid (attention block 2240, mamba page padded +0.54 % to
  match), so `max_page_size` is a *padded* value. `unify_kv_cache_spec_page_size`
  scales the drafter's attention block by `ratio` and then asserts
  `page == max` — which fails, because `page_size_bytes` ignores `block_size`
  once `page_size_padded` is set.
- **Fix** ([`patch_unify2.py`](../runtime/patch_unify2.py)): keep the *scaled*
  `block_size` **and** pad the <1 % remainder (mirrors vLLM's own
  `HiddenStateCacheSpec` handling). The earlier "pad but keep block_size=16"
  patch mis-strided the drafter KV → acceptance collapsed to 1.47 (a real bug,
  not a quant mismatch).
- `--mamba-block-size 256` (from other Spark recipes) **breaks** the 122B: it
  makes the mamba group block ≠ cache block, tripping the coordinator hash
  assert. Omit it.
- `--no-enable-prefix-caching` routes to `KVCacheCoordinatorNoPrefixCache`, which
  has no line-504 hash assert. Prefix caching is irrelevant at c=1 anyway.

Working stack: `patch_unify2` + prefix-off + INT4 (bf16 KV) target + drafter
pinned to `FLASH_ATTN`. **No FA4 shim needed** — vLLM gates FA4 to cap families
90/100/110 (excludes 120), so the drafter runs FA2. (The whole fa4-sm120 saga is
SGLang-only; SGLang's DFlash works too but its sm121 base decode is ~2× slower
than vLLM's, so it loses on absolute throughput.)

## 2. DFlash vs MTP — acceptance is task-dependent

MTP-2 (the native head) drafts 2 tokens **sequentially**, so acceptance caps at
~3. DFlash block-drafts 12 in **one parallel forward**, so its acceptance fills
the block on predictable traffic. Same harness, unpatched, flash_attn:

| Workload | accept (MTP-2 / DFlash) | tok/s (MTP-2 / DFlash) |
|---|---|---|
| Prose | 2.24 / 2.3 | 33.7 / 33.2 *(tie; use DFlash n=4)* |
| Code | 2.77 / 5.4 | 40.5 / 54.5 |
| Counting | **3.00 (maxed)** / 11 | 43.7 / 124.5 |
| Hermes (real) | 2.88 / **8.66** | 39.9 / **~81** |

The "DFlash caps at 2.3 / 33 tok/s" story was a **prose-benchmark artifact**.
On agent/code traffic DFlash pulls ~2× ahead because MTP is acceptance-saturated.
`n` (num_speculative_tokens) is task-dependent: prose → 4, agent/code → 12+.

## 3. Methodology — two non-comparable harnesses

- `bench_decode.py` = **decode-only** tok/s (excludes TTFT). Good for c=1 kernel
  comparisons. This is the ~81 Hermes number.
- `bench_albond.py` = **end-to-end** (completion_tokens / total wallclock incl.
  prefill, non-streaming, 5 prompts, run-1 discarded). This reproduces albond's
  own method and is directly comparable to his published **51.58**.

Apples-to-apples (albond's method): **DFlash n=12 unpatched = 53.7 tok/s
cross-prompt mean — already above his fully-patched MTP stack (51.58).**

## 4. The dense-bandwidth levers and the amortization law

albond's non-MTP wins are *always-on* bandwidth cuts. We ported the two that
transfer to vLLM 0.23 + DFlash:

- **hybrid INT4+FP8** ([`patch_inc_hybrid.py`](../runtime/patch_inc_hybrid.py)):
  the Intel base already stores **attention as INT4** (0.5 B/param, *better* than
  albond's FP8 attention), so the only thing to gain is the BF16 **shared
  experts** → calibrated FP8 (144 layers, 0.48 GB saved). The dispatch patch adds
  an `INCConfig.maybe_update_config` override (AEON 0.23's hook signature takes
  `hf_config=`, unlike albond's 0.19) that detects FP8 dense layers and
  dispatches `Fp8LinearMethod` for them.
- **int8 lm-head** ([`patch_int8_lmhead_v3.py`](../runtime/patch_int8_lmhead_v3.py)):
  the 248 320-row vocab projection is the single largest dense read (1.5 GB BF16,
  *every token*). A batched int8 w8a16 Triton GEMV reads it at ~227 GB/s (vs bf16
  ~6.5–8.8 ms) — **~2× faster, argmax-exact**. Prior ports failed not on the
  kernel but on **integration**: zeroing the lm-head weight corrupted the
  *drafter-shared* head (garbage), and a per-row loop for B>4 was slower under
  spec. v3 uses one batched kernel and **keeps** the bf16 weight.

Why the denominator matters: the 0.48 GB shared-expert saving is **0.7 % of the
71 GB on disk** but **~8 % of the ~6 GB *active per-token* footprint** (the disk
is mostly sparse routed experts). Shared experts and the lm-head are **dense —
read every token** — so at base decode the savings land in full:

| Config | base (acc 1) | albond-bench (acc 6.4) | Hermes (acc 8.3) |
|---|---|---|---|
| INT4 baseline | 28.2 | 53.7 | ~81 |
| + hybrid-FP8 | 30.4 (+7.8%) | 57.0 (+6.1%) | ~80 |
| + int8 lm-head | 32.7 (+16%) | — | — |
| **+ both** | **36.0 (+28%)** | **59.0 (+10%)** | ~80–87 (noise) |

The levers compose additively (step savings 2.6 + 4.9 ≈ 7.7 ms). But the uplift
**decays monotonically with acceptance**:

> Under speculative decode the verify forward processes ~`accept` positions and
> reads each dense weight **once**, amortized across them. So a dense-weight cut
> that is +X % at base is ~+X/accept % under spec.

```
+28%  base (accept 1)  →  +10%  albond-bench (accept 6.4)  →  ~0%  Hermes (accept 8.3)
```

**Consequences**

- For the **agent path** (`dflash`), DFlash's own high acceptance already
  amortizes the dense levers to ~null. Its remaining bottleneck is **routed-expert
  verify-batch reads** (each of ~13 verify positions routes to different experts)
  — untouched by any dense-weight quant. To push Hermes further you must attack
  *that*: a smaller/faster drafter, lower `n` at equal acceptance, or sub-INT4
  routed experts.
- For **base / low-acceptance** serving (`dense`), the stack is a real **+28 %**
  (36 tok/s) and is the recommended config there.

## 5. Things that did NOT help c=1 decode

- **FLASHINFER target backend** — null both short-context and Hermes (attention
  isn't the bottleneck on this GDN/mamba-heavy MoE; most layers are linear
  attention). albond's "+16 %" was on his dense-attention MTP path.
- **b12x / native FP4 MoE** — null at c=1 (a throughput/concurrency lever, not a
  latency one; at batch 1 the active-expert GEMM is tiny).
- **FLA sm121 big-tile shmem fix** — real bug, but prefill/TTFT only; c=1 decode
  uses the GDN *recurrent* path, a different kernel. Kept (free TTFT win).
- **PR#38325 swapAB FP8 GEMM** — marginal (+0.76 %), only with the FP8 checkpoint.

## Production recommendation

Ship **`dflash`** for the agent (DFlash unpatched, ~81 tok/s, ~2× MTP, > albond's
patched 51.58). Reserve **`dense`** for base / low-acceptance serving (+28 %).
The dense patches are upside there, not a requirement for the agent to win.
