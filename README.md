# qwen3.5-122B-A10B-on-spark

[`Qwen3.5-122B-A10B`](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound)
(hybrid GDN + mamba + 128-expert MoE, ~10B active) running on a single
**NVIDIA DGX Spark** (GB10 / SM121, 128 GB / 119 GiB unified) under **vLLM**, with
**[DFlash](https://modal.com/blog/spec-is-all-u-need) block-diffusion speculative
decode** and an optional **dense-bandwidth patch stack** — measured end-to-end,
with a per-token bandwidth model that explains every number.

**Status:** Working end-to-end, one-shot install. On real Hermes-agent
tool-call turns, **DFlash decode reaches a median ~81 tok/s on GB10** —
**~2× the native MTP-2 head (~40 tok/s)** on the same workload, and above
[**albond's**](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)
fully-patched MTP recipe (51.58 tok/s end-to-end) — **the recipe this repo
gratefully builds on** (see [Building on the albond recipe](#building-on-the-albond-recipe)).
DFlash's acceptance is task-dependent (it block-drafts 12 tokens in one parallel
forward), so the win is largest on structured/tool-call/code traffic and
collapses to parity on open-ended prose.

A separate **dense-bandwidth stack** (hybrid INT4+FP8 shared experts + int8
lm-head) adds **+28 % to no-spec / base decode** (28.2 → 36.0 tok/s) but, by the
amortization law below, washes out to ~null on high-acceptance agent traffic —
so it's a lever for *base / low-acceptance* serving, not for the agent path.

- **Engine:** [`vLLM`](https://github.com/vllm-project/vllm) 0.23, sm121 build with the DFlash PRs, via the prebuilt image `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`. No host build — the four runtime patches in [`runtime/`](runtime/) are applied at serve time.
- **Target:** [`Intel/Qwen3.5-122B-A10B-int4-AutoRound`](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) — INT4 (AutoRound/GPTQ) routed experts + attention, BF16 shared experts/embeddings/head, ~62 GiB. (Safetensors, *not* GGUF — vLLM serves HF checkpoints directly.)
- **Drafter:** [`z-lab/Qwen3.5-122B-A10B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) — 0.8B / 6-layer non-causal block-diffusion drafter (block 16), shares the target's `embed_tokens` + `lm_head`, ~1.6 GiB.
- **Hardware:** NVIDIA DGX Spark, GB10, SM121, 128 GB LPDDR5X unified (~119 GiB usable), ~273 GB/s.

## Quick start

On a DGX Spark with Docker + the NVIDIA container runtime:

```bash
curl -sSL https://raw.githubusercontent.com/Entrpi/qwen3.5-122B-A10B-on-spark/main/install.sh | bash -s -- --start
```

That one command:

1. Verifies the host (aarch64, GB10/SM121, Docker GPU access, free disk).
2. Pulls the sm121 vLLM image (~40 GiB, one-time).
3. Downloads the INT4 target (~62 GiB) + DFlash drafter (~1.6 GiB) into the HF cache.
4. Starts the `dflash` profile on `:8000`, waits until READY, and runs the
   "capital of France" smoke test (asserts "Paris").

**Already have the model?** Skip the 62 GiB download:

```bash
# point at a checkpoint dir you already have (mounted read-only at /model):
./install.sh --start --model-dir /path/to/Qwen3.5-122B-A10B-int4-AutoRound
# or reuse an existing HF cache (download becomes a no-op if already present):
./install.sh --start --hf-home /mnt/big/hf
```

Preview without running: `... | bash -s -- --help`.

## Hardware requirements

| | |
|---|---|
| Validated on | NVIDIA DGX Spark (GB10, SM121, 128 GB / 119 GiB unified) |
| Likely to work | other Blackwell with `--force` (untested) |
| Runtime | Docker + NVIDIA container runtime (`docker run --gpus all`) |
| Disk | ≥ 75 GiB free (image + weights); ≥ 150 GiB if `--build-hybrid` |
| OS | aarch64 Linux (Grace) |
| Memory | 128 GB / 119 GiB unified is enough for the model + DFlash drafter + KV @ 16k |

GB10 is detected via `nvidia-smi --query-gpu=compute_cap` returning `12.1`;
anything else needs `--force`.

### Memory & context (defaults are tuned for max context + KV depth)

This model is **36/48 linear-attention (GDN) + 12 full-attention** layers, and the
attention layers use GQA `num_key_value_heads=2`, `head_dim=256` → **only
~24 KiB/token of KV**. A full **262 144** context (the model's native max) is just
~6 GiB of KV; the GDN layers hold a *fixed* per-sequence state (~0.2 GiB) that
does **not** grow with context. So on the 128 GB (119 GiB) GB10, reserving 14 GB
for the OS leaves ~106 GiB for vLLM:

| | |
|---|---|
| weights (INT4) + DFlash drafter | ~64 GiB |
| CUDA graphs + activations | ~10 GiB |
| **KV pool** | **~32 GiB ≈ 1.38 M tokens** |

The KV pool (~1.38 M tokens) dwarfs a single 262 144 context, so single context is
capped by the *model*, not memory. Defaults (override via flags/env):

| Flag / env | Default | Note |
|---|---|---|
| `--gpu-mem` / `GPU_MEM` | **0.89** | reserves ~14 GB; drop to 0.87 if the OOM-guard fires on first load |
| `--ctx` / `CTX` (`MAX_MODEL_LEN`) | **262144** | model native max |
| `--max-num-seqs` / `MAX_NUM_SEQS` | **1** | single-stream; raising it is nearly free (pool ≫ one context) |
| `--max-batched-tokens` / `MAX_BATCHED_TOKENS` | **8192** | chunked-prefill chunk — kept **below** ctx so a long prefill doesn't batch all at once |

> Unified-memory OOM **hard-freezes** the box, and vLLM's profiler can undershoot
> peak by a couple GB — always bring the server up under
> [`scripts/monitor.sh`](scripts/monitor.sh) (OOM auto-kill guard) the first time
> at a new `gpu-mem`/`ctx`.

## What you get — profiles

Pick with `--profile`:

| Profile | Stack | Best for | Measured |
|---|---|---|---|
| **`dflash`** *(default)* | INT4 + DFlash n=12 | agents / tool-calls / code | **~81 tok/s** Hermes · 53.7 albond-bench |
| `dense` | hybrid INT4+FP8 + int8 lm-head + DFlash n=12 | base / low-accept serving | 36.0 base (+28%) · 59.0 albond-bench |
| `base` | plain INT4, no spec | airtight baseline | 28.2 tok/s c=1 |
| `mtp` | INT4 + native MTP-2 head | comparison | ~40 tok/s Hermes |

The server is OpenAI-compatible (`/v1/chat/completions` with tool calls + SSE,
`/v1/completions`, `/v1/models`) and serves under the model name `qwen`.

## Benchmarks

All single-stream (c=1), temperature 0, GB10. "Hermes" = regenerating the next
assistant turn over 10 real conversations from a live agent's `state.db` (73 %
tool-calls); "albond-bench" = albond's own end-to-end harness (completion_tokens
/ total wallclock incl. prefill, 5 prompts, run-1 discarded — directly
comparable to his published 51.58).

### DFlash vs MTP, same harness, unpatched

| Workload (accept len) | base no-spec | MTP-2 | **DFlash n=12** |
|---|---|---|---|
| Prose (~2.3) | 28.2 | 33.7 | 33.2 *(use n=4)* |
| Code (~5.4) | 28.2 | 40.5 | **54.5** |
| Counting (~11) | 28.2 | 43.7 *(MTP caps at acc 3)* | **124.5** |
| **Hermes, real turns (8.3)** | — | **39.9** | **~81** |
| albond-bench e2e (6.5) | — | — | **53.7** |

MTP-2 drafts 2 tokens *sequentially* (acceptance caps at ~3); DFlash block-drafts
12 in **one parallel forward**, so on predictable/agent traffic it accepts 5–11
and pulls ~2× ahead. They tie only on low-acceptance prose. **53.7 unpatched
already clears albond's fully-patched MTP (51.58)** under his own method.

### The dense-bandwidth stack (`dense` profile)

Two independent always-on levers, ported to vLLM 0.23 as runtime patches:
hybrid INT4+FP8 (BF16 shared experts → calibrated FP8) and int8 lm-head (the
248 320-row vocab projection → int8 w8a16 GEMV, ~2× the bf16 read).

| Config | base (acc 1) | DFlash spec, albond-bench (acc 6.4) | Hermes (acc 8.3) |
|---|---|---|---|
| INT4 baseline | 28.2 | 53.7 | ~81 |
| + hybrid-FP8 | 30.4 (+7.8%) | 57.0 (+6.1%) | ~80 |
| + int8 lm-head | 32.7 (+16%) | — | — |
| **+ both** | **36.0 (+28%)** | **59.0 (+10%)** | ~80–87 *(noise)* |

## Building on the albond recipe

This repo stands on **[albond's DGX-Spark Qwen3.5-122B recipe](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)** —
the first working high-throughput recipe for this model on Spark, and the
reference we measured everything against. On eugr's vLLM 0.19.1 fork, albond
established:

- the **rebuilt hybrid INT4+FP8 checkpoint** (BF16 dense → calibrated FP8),
- the **INT8 lm-head** patch (the single biggest dense-bandwidth lever),
- **MTP-2** native speculative decode, and
- the **end-to-end benchmark methodology** (completion_tokens / total wallclock,
  incl. prefill) we report against — reproduced verbatim in
  [`scripts/bench_albond.py`](scripts/bench_albond.py).

We gratefully build on all of it. What this repo adds is **carrying that recipe
forward to the latest vLLM and a stronger drafter**, and getting the community
patches to stack together there:

1. **Forward-ported the dense levers to vLLM 0.23.** albond's patches target the
   0.19 fork and don't drop in cleanly. The hybrid-FP8 dispatch had to be
   re-expressed against 0.23's `maybe_update_config(model_name, hf_config=…)`
   quant-config hook; the INT8 lm-head needed a from-scratch integration (the
   prior port zeroed the lm-head weight — which corrupts the **DFlash-shared**
   head → garbage — and looped per-row for batch>4, which is *slower* under spec;
   both fixed in [`patch_int8_lmhead_v3.py`](runtime/patch_int8_lmhead_v3.py)).
2. **Swapped MTP-2 for the DFlash block-diffusion drafter** and got it running on
   the hybrid 122B in vLLM (the [KV-unify fix](runtime/patch_unify2.py)). DFlash
   block-drafts 12 tokens in one parallel forward vs MTP's sequential head
   (acceptance-capped at ~3), so it pulls ~2× ahead on agent/code traffic.
3. **Composed the dense levers *with* DFlash** instead of MTP.

### Where we land — albond's own end-to-end method, same hardware class

| Stack | Spec | Dense patches | e2e tok/s |
|---|---|---|---|
| **albond** (published) | MTP-2 | hybrid-FP8 + INT8 lm-head + PR#38325 | 51.58 |
| this repo — `dflash` | DFlash n=12 | *none* | **53.7**  (+4%) |
| this repo — `dense` | DFlash n=12 | hybrid-FP8 + INT8 lm-head | **59.0**  (+14%) |

On the **real agent workload** (decode-only, regenerating live tool-call turns),
DFlash's parallel block-drafting pulls further ahead of MTP's sequential head:
**~81 vs ~40 tok/s**.

> **Stated plainly:** albond's 51.58 is his published figure on his stack
> (vLLM 0.19 + MTP); our figures are on this stack (vLLM 0.23 + DFlash). Both use
> the **same e2e harness** on the **same hardware class** (DGX Spark / GB10) — a
> fair best-on-each-stack comparison, not a single-variable controlled run. Note
> that **unpatched DFlash (53.7) already clears albond's fully-patched MTP**, so
> the dense stack is upside on top of the drafter swap, not the source of the win.

## The amortization law

The dense levers cut **always-on** weight reads (shared experts + lm-head, read
every token). Under speculative decode the verify forward reads those weights
**once and amortizes them across the accepted block**, so the gain shrinks as
acceptance rises — monotonically, across the whole curve:

```
dense stack uplift:   +28%  (base, accept 1)
                  →   +10%  (albond-bench, accept ~6.4)
                  →   ~0%   (Hermes, accept ~8.3)
```

**Consequence:** for the **agent path (`dflash`)**, DFlash's own high acceptance
already saturates the dense levers — its remaining bottleneck is *routed-expert*
verify-batch reads, which no dense-weight quant touches. For **base / low-accept
serving (`dense`)**, the stack is a real +28 %. See [`docs/FINDINGS.md`](docs/FINDINGS.md).

## Under the hood: the four runtime patches

vLLM is unmodified on disk; [`runtime/serve.sh`](runtime/serve.sh) edits the
installed package in-place before `vllm serve` (idempotent, sentinel-guarded):

| Patch | What it does | Needed by |
|---|---|---|
| [`patch_unify2.py`](runtime/patch_unify2.py) | scale-block KV-cache **unify** so the hybrid GDN+mamba target absorbs the drafter's attention spec (the original assert can't); + `--no-enable-prefix-caching` routes to the no-hash-assert coordinator | **DFlash** (any spec profile) |
| [`patch_inc_hybrid.py`](runtime/patch_inc_hybrid.py) | adds an `INCConfig.maybe_update_config` override that detects FP8 dense layers in the hybrid checkpoint and dispatches `Fp8LinearMethod` for `shared_expert` | `dense` |
| [`patch_int8_lmhead_v3.py`](runtime/patch_int8_lmhead_v3.py) | replaces the lm-head matmul in `_get_logits` with a batched int8 w8a16 Triton GEMV (keeps bf16 weight for the shared drafter) | `dense` |
| [`patch_fla_shmem.py`](runtime/patch_fla_shmem.py) | lets the FLA GDN chunk kernels use big tiles on sm121's 99 KiB shmem (prefill/TTFT only; harmless) | always (free) |

Why DFlash needs the unify patch at all, why the drafter must run `FLASH_ATTN`
(non-causal), and the full vLLM-vs-SGLANG dead-end history are in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## Repo layout

```
install.sh                 One-shot installer (curl | bash | --help)
runtime/                   Mounted read-only at /host inside the container:
  serve.sh                   vLLM serve wrapper (applies the patches, then serves)
  patch_unify2.py            DFlash KV-unify fix
  patch_inc_hybrid.py        hybrid INT4+FP8 dispatch
  patch_int8_lmhead_v3.py    int8 lm-head GEMV
  patch_fla_shmem.py         FLA sm121 big-tile (prefill)
  mtp_serve.sh               MTP-2 comparison serve
scripts/                   Host-side helpers:
  monitor.sh                 Container-startup monitor with OOM auto-kill guard
  bench_decode.py            Decode-only tok/s (excludes TTFT)
  bench_albond.py            albond's e2e method (comparable to his 51.58)
  hermes_bench.py            Real agent turns from ~/.hermes/state.db
  run_bank.sh                prose/code/counting/hermes bank on any server
tools/
  build-hybrid-checkpoint.py Build the hybrid INT4+FP8 ckpt (for --build-hybrid)
  inspect_ckpt.py            Which layers are INT4 vs BF16 vs FP8
  validate_*.py              Standalone correctness checks for the patches
docs/
  FINDINGS.md                The full investigation, methodology, and the
                             amortization-law derivation
```

## Reproducing

```bash
# default agent path (DFlash) + smoke test:
./install.sh --start

# the dense stack (build the hybrid ckpt once, ~20 min, then serve):
./install.sh --build-hybrid
./install.sh --start --profile dense

# benches (run on the host against the server; need: pip install requests):
python3 scripts/bench_decode.py  --base-url http://127.0.0.1:8000 --model qwen \
        --prompt "Write a detailed essay about the history of tea."
python3 scripts/bench_albond.py  http://127.0.0.1:8000 "dflash"     # e2e, vs 51.58
python3 scripts/hermes_bench.py  --base-url http://127.0.0.1:8000   # real agent turns

# MTP comparison:
./install.sh --start --profile mtp
```

## How this fits with related work

| Piece | Role |
|---|---|
| [`vLLM`](https://github.com/vllm-project/vllm) | the inference engine; this repo serves Qwen3.5 + DFlash on it, unmodified-on-disk |
| [`Intel/...int4-AutoRound`](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) · [`z-lab/...DFlash`](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) | the target + drafter weights |
| [`albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4`](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) | the MTP + hybrid-FP8 + int8-lmhead recipe we benchmarked against and ported the dense levers from |
| [`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) | sibling repo, same hardware, different model (DeepSeek-V4-Flash via ds4) |
| [Modal: *Speculative decoding is all you need*](https://modal.com/blog/spec-is-all-u-need) | the DFlash block-diffusion drafter and the task-dependent-acceptance framing |

## Acknowledgements

- **[`albond`](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) — the foundation this builds on.** The hybrid INT4+FP8 checkpoint, the INT8 lm-head, MTP-2, and the end-to-end benchmark methodology. This repo is a grateful forward-port and recomposition of that recipe onto vLLM 0.23 + DFlash.
- [`z-lab`](https://huggingface.co/z-lab) / [Modal](https://modal.com/blog/spec-is-all-u-need) — the DFlash drafter and the block-diffusion speculative-decode work.
- [`Intel/AutoRound`](https://huggingface.co/Intel) — the INT4 target quantization.
- [`vLLM`](https://github.com/vllm-project/vllm) and the AEON sm121 image maintainers — the engine and the DFlash-enabled GB10 build.
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) — the vLLM-on-Spark base that albond's recipe (and much of this ecosystem) started from.

## License

MIT — see [LICENSE](LICENSE). The patches are original; vendored third-party
files (`tools/build-hybrid-checkpoint.py`) retain their upstream attribution.
