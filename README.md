# qwen3.5-122B-A10B-on-spark

[`Qwen3.5-122B-A10B`](https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid)
(hybrid GDN + mamba + 128-expert MoE, ~10B active) on a single
**NVIDIA DGX Spark** (GB10 / SM121, 128 GB / 119 GiB unified) under **vLLM**, with
**[DFlash](https://modal.com/blog/spec-is-all-u-need) block-diffusion speculative
decode** and a **dense-bandwidth patch stack** — measured end-to-end, with a
per-token bandwidth model behind the numbers.

**The default `dense` profile** serves a purpose-built **hybrid INT4+FP8
checkpoint**, [`bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid`](https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid),
downloaded ready-to-run — no local checkpoint build.

**Status:** working end-to-end, one-shot install. On real agent tool-call turns,
**DFlash decode reaches a median ~81 tok/s on GB10** — about **2× the native
MTP-2 head (~40 tok/s)** on the same workload, and above
[**albond's**](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)
fully-patched MTP recipe (51.58 tok/s end-to-end), the recipe this project builds
on (see [Building on the albond recipe](#building-on-the-albond-recipe)). DFlash
acceptance is task-dependent (it block-drafts 12 tokens in one parallel forward),
so the gain is largest on structured / tool-call / code traffic and falls to
parity on open-ended prose.

The default **`dense`** profile bundles a **dense-bandwidth stack** (hybrid
INT4+FP8 shared experts + int8 lm-head) on top of DFlash, downloaded as a prebuilt
checkpoint: **+28 % on no-spec / base decode** (28.2 → 36.0 tok/s) and equal to
plain DFlash on high-acceptance agent traffic (per the
[amortization law](#the-amortization-law) below), at a modestly smaller KV pool.
The **`dflash`** profile drops the dense patches for the largest KV pool and the
same ~81 tok/s on agents.

- **Model (default):** [`bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid`](https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid) — the prebuilt hybrid the `dense` profile downloads and serves: INT4 (AutoRound/GPTQ) routed experts + attention, **calibrated FP8** shared experts, BF16 embeddings / head; ~67 GiB safetensors (*not* GGUF — vLLM serves it directly). Built from [`Intel/Qwen3.5-122B-A10B-int4-AutoRound`](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) (INT4 base) + [`Qwen/Qwen3.5-122B-A10B-FP8`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) (FP8 shared experts). The `dflash` / `base` / `mtp` profiles serve the plain `Intel/…int4-AutoRound` instead.
- **Drafter:** [`z-lab/Qwen3.5-122B-A10B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) — 0.8B / 6-layer non-causal block-diffusion drafter (block 16), sharing the model's `embed_tokens` + `lm_head`, ~1.6 GiB.
- **Engine:** [`vLLM`](https://github.com/vllm-project/vllm) 0.23, sm121 build with the DFlash PRs, via the prebuilt image `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`. No host build — the runtime patches in [`runtime/`](runtime/) are applied at serve time.
- **Hardware:** NVIDIA DGX Spark, GB10, SM121, 128 GB LPDDR5X unified (~119 GiB usable), ~273 GB/s.

## Quick start

On a DGX Spark with Docker and the NVIDIA container runtime:

```bash
curl -sSL https://raw.githubusercontent.com/Entrpi/qwen3.5-122B-A10B-on-spark/main/install.sh | bash -s -- --start
```

This command:

1. Verifies the host (aarch64, GB10 / SM121, Docker GPU access, free disk).
2. Pulls the sm121 vLLM image (~40 GiB, one-time).
3. Downloads the hybrid checkpoint (~67 GiB) and the DFlash drafter (~1.6 GiB) into the HF cache.
4. Starts the **`dense`** profile on `:8000`, waits until READY (~3 min), and runs
   the "capital of France" smoke test (asserts "Paris").

For the plain-DFlash agent profile (largest KV pool, no dense patches, serves the
Intel INT4 checkpoint):

```bash
./install.sh --start --profile dflash
```

To reuse a checkpoint already on disk and skip the download:

```bash
# an existing checkpoint directory (mounted read-only at /model):
./install.sh --start --model-dir /path/to/checkpoint
# or reuse an existing HF cache (download becomes a no-op if already present):
./install.sh --start --hf-home /mnt/big/hf
```

Preview without running: append `--help`.

## Hardware requirements

| | |
|---|---|
| Validated on | NVIDIA DGX Spark (GB10, SM121, 128 GB / 119 GiB unified) |
| Likely to work | other Blackwell with `--force` (untested) |
| Runtime | Docker + NVIDIA container runtime (`docker run --gpus all`) |
| Disk | ≥ 75 GiB free (image + weights); ≥ 150 GiB with `--build-hybrid` |
| OS | aarch64 Linux (Grace) |
| Memory | 128 GB / 119 GiB unified holds the model + DFlash drafter + KV |

GB10 is detected via `nvidia-smi --query-gpu=compute_cap` returning `12.1`; other
hardware requires `--force`.

### Memory and context (defaults tuned for up to 3 concurrent streams)

The intended deployment is a single user with up to **3 concurrent decode
streams** (for example, a main agent thread plus subagents). A single stream can
use the full **262 144** context; the defaults assume concurrent streams stay
under ~100 k tokens each. The default operating point is one stream with no
contention; two to three streams are additive (see the per-workload table below).

Attention KV is small for this model (~24 KiB/token — 12 of 48 layers are full
attention, with GQA `num_key_value_heads=2`), but the per-sequence GDN/mamba state
and the activation reserve dominate the footprint, so the usable pool is far
smaller than a KV-only estimate suggests. The values below are measured on the
target hardware at the shipped defaults (`gpu-mem 0.82`, `ctx 262144`, `seqs 3`):

| Measurement (default `dense` profile) | Value |
|---|---|
| Free memory at READY | **~16 GiB** (responsive, no swap; ~15 GiB under peak 3-stream load) |
| GPU KV cache pool | **426,610 tokens** (`dflash` profile: 456,664) |
| Max concurrency at full 262 144 | **1.63×** (`dflash`: 1.74×) |

The `dense` pool was lifted from 376,518 → **426,610 tokens** (+13 %) at the *same*
`0.82` headroom by reclaiming over-reserved memory rather than spending headroom:
the int8 lm-head frees its now-dead bf16 copy (~1.4 GiB — the DFlash drafter shares
the same int8 lm_head, and `tie_word_embeddings=False`, so it is genuinely unused
after quantization), and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` returns the
CUDA-graph over-estimate (~0.6 GiB; actual capture is ~0.14 GiB, drawn from the wide
0.82 headroom). Both are validated coherent with full drafter acceptance (4–12).

Decode-only throughput (streaming, excludes prefill), by workload and concurrency:

| Workload | 1 stream | 2 streams | 3 streams | aggregate @ 3 |
|---|---|---|---|---|
| prose | 48.8 | 38.4 | 30.8 | 92.5 |
| code | 66.9 | 55.3 | 43.0 | 129 |
| agentic (real 6 k tool-call ctx) | 121.8 | 98.9 | 66.7 | 200 |

(tok/s per stream; aggregate is the sum across streams. Each added stream lowers
per-stream throughput and raises the aggregate. The single-context agentic 121.8
exceeds the ~81 headline, which is the median over 10 varied real turns including
longer, slower-prefilling contexts.)

A typical load — three streams under ~100 k each (≈ <300 k tokens) — fits the
426 k pool with margin, and a single stream can still reach the full 262 144
context. At `gpu-mem` 0.88–0.89 the static footprint leaves only ~5 GiB free; the
host then swaps and requests stall. `0.82` is the validated value (~16 GiB free
on `dense`, ~15 GiB under peak load). Defaults (override via flags or environment variables):

| Flag / env | Default | Note |
|---|---|---|
| `--gpu-mem` / `GPU_MEM` | **0.82** | ~14 GiB free (validated); 0.88+ over-subscribes and swaps |
| `--ctx` / `CTX` (`MAX_MODEL_LEN`) | **262144** | model native max; a single stream can reach any length up to this. Costs only KV-pool sizing — the CUDA-graph compile range tracks `max-batched-tokens`, not `ctx` |
| `--max-num-seqs` / `MAX_NUM_SEQS` | **3** | concurrent-stream cap; the pool holds ~1.6× (`dense`) / ~1.7× (`dflash`) a full-262 k context, ample for <100 k streams |
| `--max-batched-tokens` / `MAX_BATCHED_TOKENS` | **8192** | chunked-prefill chunk, kept **below** `ctx` so a long prefill does not batch all at once |

The default operating point is a single stream (no contention; ~81 tok/s on agent
turns). Two to three concurrent streams are additive: per-stream throughput
decreases as the decode batch grows (more routed-expert traffic per step) while
aggregate throughput rises — a throughput-versus-latency trade, not a safety
limit. A fourth simultaneous stream requires `--max-num-seqs 4` (additional
streams queue rather than fail) or a smaller `--ctx`.

> Unified-memory OOM hard-freezes the host, and the vLLM profiler can undershoot
> peak by a couple of GB. Bring the server up under
> [`scripts/monitor.sh`](scripts/monitor.sh) (OOM auto-kill guard) the first time
> at any new `gpu-mem` or `ctx`.

### Startup time

Time to READY is **~3 min** (vs ~12 min on the default loader). `--load-format
fastsafetensors` (default) reads the 62–71 GiB **straight to the device** —
**~32 s** vs ~8 min for the default mmap-backed per-tensor read + redundant
CPU→GPU copy, which is pathologically slow on GB10's current kernel (the copy is
also physically pointless on unified memory). It falls back to `nogds`
automatically — no GPUDirect Storage hardware required; override with
`LOAD_FORMAT=auto`. The remaining ~2.5 min is `torch.compile` (~42 s) + CUDA-graph
capture (~2 min); the compile cache is persisted at `$HF_HOME/.vllm_cache`, so
boots after the first skip the compile.

## Profiles

Selected with `--profile`:

| Profile | Stack | Best for | Measured |
|---|---|---|---|
| **`dense`** *(default)* | hybrid INT4+FP8 + int8 lm-head + DFlash n=12 | general — downloads the prebuilt hybrid; ≈ dflash on agents, +28% on base | 36.0 base (+28%) · 59.0 albond-bench · ~81 Hermes |
| `dflash` | INT4 + DFlash n=12 | agent path; largest KV pool (456k vs 426k) | **~81 tok/s** Hermes · 53.7 albond-bench |
| `base` | plain INT4, no speculative decode | airtight baseline | 28.2 tok/s c=1 |
| `mtp` | INT4 + native MTP-2 head | comparison | ~40 tok/s Hermes |

The server is OpenAI-compatible (`/v1/chat/completions` with tool calls + SSE,
`/v1/completions`, `/v1/models`) and serves under the model name `qwen`.

**Tool calling is enabled by default** (`--enable-auto-tool-choice
--tool-call-parser qwen3_xml --reasoning-parser qwen3`), so clients can send `tools`
with `tool_choice="auto"`. Qwen3.5 emits the *XML* tool format, so `qwen3_xml` is the
correct parser — `hermes` returns 200 but with empty `tool_calls` (the call lands in
`content` instead). `<think>` reasoning is split into `reasoning_content`. Override
with `TOOL_PARSER=` / `REASONING_PARSER=` (empty disables; `qwen3_coder` also works).

**Network:** the server binds `0.0.0.0` in `--net=host`, so it is reachable from the
LAN at `http://<spark-ip>:8000` — the `127.0.0.1` in the examples is just the local
default. There is no auth; put it behind a reverse proxy / firewall for shared use.

## Benchmarks

All single-stream (c=1), temperature 0, GB10. "Hermes" regenerates the next
assistant turn over 10 real conversations from a live agent's `state.db` (73 %
tool-calls); "albond-bench" is albond's end-to-end harness (completion tokens /
total wallclock incl. prefill, 5 prompts, run 1 discarded — directly comparable to
the published 51.58).

### DFlash vs MTP, same harness, unpatched

| Workload (accept len) | base no-spec | MTP-2 | **DFlash n=12** |
|---|---|---|---|
| Prose (~2.3) | 28.2 | 33.7 | 33.2 *(use n=4)* |
| Code (~5.4) | 28.2 | 40.5 | **54.5** |
| Counting (~11) | 28.2 | 43.7 *(MTP caps at acc 3)* | **124.5** |
| **Hermes, real turns (8.3)** | — | **39.9** | **~81** |
| albond-bench e2e (6.5) | — | — | **53.7** |

MTP-2 drafts 2 tokens *sequentially*, so acceptance caps at ~3; DFlash block-drafts
12 in **one parallel forward**, accepting 5–11 on predictable / agent traffic and
running ~2× ahead. The two tie only on low-acceptance prose. Unpatched DFlash
(53.7) already clears albond's fully-patched MTP (51.58) under the same end-to-end
method.

### Dense-bandwidth stack (`dense` profile)

Two independent always-on levers, ported to vLLM 0.23 as runtime patches: hybrid
INT4+FP8 (BF16 shared experts → calibrated FP8) and int8 lm-head (the 248 320-row
vocab projection → int8 w8a16 GEMV, ~2× the bf16 read).

| Config | base (acc 1) | DFlash spec, albond-bench (acc 6.4) | Hermes (acc 8.3) |
|---|---|---|---|
| INT4 baseline | 28.2 | 53.7 | ~81 |
| + hybrid-FP8 | 30.4 (+7.8%) | 57.0 (+6.1%) | ~80 |
| + int8 lm-head | 32.7 (+16%) | — | — |
| **+ both** | **36.0 (+28%)** | **59.0 (+10%)** | ~80–87 *(noise)* |

## Building on the albond recipe

This project builds on
**[albond's DGX-Spark Qwen3.5-122B recipe](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)** —
the first working high-throughput recipe for this model on Spark, and the
reference used for the comparisons here. On eugr's vLLM 0.19.1 fork, albond
established:

- the **rebuilt hybrid INT4+FP8 checkpoint** (BF16 dense → calibrated FP8),
- the **INT8 lm-head** patch (the single largest dense-bandwidth lever),
- **MTP-2** native speculative decode, and
- the **end-to-end benchmark methodology** (completion tokens / total wallclock,
  incl. prefill), reproduced verbatim in [`scripts/bench_albond.py`](scripts/bench_albond.py).

This project carries that recipe forward to the latest vLLM and a stronger
drafter, and composes the community patches there:

1. **Dense levers forward-ported to vLLM 0.23.** albond's patches target the 0.19
   fork and do not apply cleanly. The hybrid-FP8 dispatch is re-expressed against
   0.23's `maybe_update_config(model_name, hf_config=…)` quant-config hook; the
   INT8 lm-head is reintegrated from scratch (the prior port zeroed the lm-head
   weight — which corrupts the **DFlash-shared** head — and looped per-row for
   batch > 4, which is slower under speculative decode; both are fixed in
   [`patch_int8_lmhead_v3.py`](runtime/patch_int8_lmhead_v3.py)).
2. **MTP-2 replaced by the DFlash block-diffusion drafter,** running on the hybrid
   122B in vLLM via the [KV-unify fix](runtime/patch_unify2.py). DFlash block-drafts
   12 tokens in one parallel forward versus MTP's sequential head (acceptance-capped
   at ~3), running ~2× ahead on agent / code traffic.
3. **Dense levers composed with DFlash** rather than MTP.

### Comparison — albond's end-to-end method, same hardware class

| Stack | Spec | Dense patches | e2e tok/s |
|---|---|---|---|
| **albond** (published) | MTP-2 | hybrid-FP8 + INT8 lm-head + PR#38325 | 51.58 |
| this project — `dflash` | DFlash n=12 | *none* | **53.7**  (+4%) |
| this project — `dense` | DFlash n=12 | hybrid-FP8 + INT8 lm-head | **59.0**  (+14%) |

On the real agent workload (decode-only, regenerating live tool-call turns),
DFlash's parallel block-drafting runs further ahead of MTP's sequential head:
**~81 vs ~40 tok/s**.

> **Note on the comparison.** The 51.58 is the published figure for albond's stack
> (vLLM 0.19 + MTP); the figures here are for this stack (vLLM 0.23 + DFlash). Both
> use the same end-to-end harness on the same hardware class (DGX Spark / GB10) — a
> best-on-each-stack comparison, not a single-variable controlled run. Unpatched
> DFlash (53.7) already exceeds the fully-patched MTP result, so the dense stack is
> additional headroom rather than the source of the difference.

## The amortization law

The dense levers cut **always-on** weight reads (shared experts and lm-head, read
every token). Under speculative decode the verify forward reads those weights
**once and amortizes them across the accepted block**, so the gain shrinks as
acceptance rises — monotonically, across the curve:

```
dense stack uplift:   +28%  (base, accept 1)
                  →   +10%  (albond-bench, accept ~6.4)
                  →   ~0%   (Hermes, accept ~8.3)
```

For the agent path (`dflash`), DFlash's high acceptance already saturates the
dense levers; the remaining bottleneck is *routed-expert* verify-batch reads,
which no dense-weight quantization touches. For base / low-acceptance serving
(`dense`), the stack is a real +28 %. Full derivation in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## Under the hood: the five runtime patches

vLLM is unmodified on disk; [`runtime/serve.sh`](runtime/serve.sh) edits the
installed package in-place before `vllm serve` (idempotent, sentinel-guarded):

| Patch | Effect | Required by |
|---|---|---|
| [`patch_unify2.py`](runtime/patch_unify2.py) | scale-block KV-cache **unify** so the hybrid GDN+mamba target absorbs the drafter's attention spec (the upstream assert cannot) | DFlash (any spec profile) |
| [`patch_prefix_align.py`](runtime/patch_prefix_align.py) | makes `resolve_kv_cache_block_sizes`' mamba back-off **align-aware** so prefix caching coexists with DFlash (uses the GCD hash block size, 2240, instead of the LCM that fails the coordinator assert) | prefix caching ON (default) |
| [`patch_inc_hybrid.py`](runtime/patch_inc_hybrid.py) | adds an `INCConfig.maybe_update_config` override that detects FP8 dense layers in the hybrid checkpoint and dispatches `Fp8LinearMethod` for `shared_expert` | `dense` |
| [`patch_int8_lmhead_v3.py`](runtime/patch_int8_lmhead_v3.py) | replaces the lm-head matmul in `_get_logits` with a batched int8 w8a16 Triton GEMV (keeps the bf16 weight for the shared drafter) | `dense` |
| [`patch_fla_shmem.py`](runtime/patch_fla_shmem.py) | allows the FLA GDN chunk kernels to use large tiles on sm121's 99 KiB shmem (prefill / TTFT only; harmless) | always (free) |

The rationale for the unify patch, the requirement that the drafter run
`FLASH_ATTN` (non-causal), and the full vLLM-vs-SGLang history are in
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
  bench_albond.py            albond's e2e method (comparable to 51.58)
  hermes_bench.py            Real agent turns from ~/.hermes/state.db
  conc_workloads.py          Per-workload concurrency sweep (decode-only)
  run_bank.sh                prose / code / counting / hermes bank
tools/
  build-hybrid-checkpoint.py Builds the hybrid INT4+FP8 checkpoint (--build-hybrid)
  inspect_ckpt.py            Reports which layers are INT4 / BF16 / FP8
  validate_*.py              Standalone correctness checks for the patches
docs/
  FINDINGS.md                Full investigation, methodology, and the
                             amortization-law derivation
```

## Reproducing

```bash
# default: the dense profile (downloads the prebuilt hybrid) + smoke test:
./install.sh --start
# (to build the hybrid checkpoint locally instead of downloading it:)
#   ./install.sh --build-hybrid && ./install.sh --start

# plain-DFlash agent path (serves the Intel INT4 checkpoint, largest KV pool):
./install.sh --start --profile dflash

# benchmarks (run on the host against the server; requires: pip install requests):
python3 scripts/bench_decode.py --base-url http://127.0.0.1:8000 --model qwen \
        --prompt "Write a detailed essay about the history of tea."
python3 scripts/bench_albond.py http://127.0.0.1:8000 "dflash"     # e2e, vs 51.58
python3 scripts/hermes_bench.py --base-url http://127.0.0.1:8000   # real agent turns
python3 scripts/conc_workloads.py --base-url http://127.0.0.1:8000 # concurrency sweep

# MTP comparison:
./install.sh --start --profile mtp
```

## Related work

| Project | Role |
|---|---|
| [`bleysg/...int4-fp8-hybrid`](https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid) | **the default served checkpoint** — this project's prebuilt hybrid INT4+FP8 |
| [`vLLM`](https://github.com/vllm-project/vllm) | the inference engine; served unmodified-on-disk |
| [`Intel/...int4-AutoRound`](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) (INT4 base) · [`Qwen/...FP8`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) (FP8 donor) · [`z-lab/...DFlash`](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) (drafter) | upstream weights the hybrid + `dflash` profiles build on |
| [`albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4`](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) | the MTP + hybrid-FP8 + int8-lm-head recipe; benchmark reference and source of the ported dense levers |
| [`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) | sibling repo, same hardware, different model (DeepSeek-V4-Flash via ds4) |
| [Modal: *Speculative decoding is all you need*](https://modal.com/blog/spec-is-all-u-need) | the DFlash block-diffusion drafter and the task-dependent-acceptance framing |

## Acknowledgements

- **[`albond`](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)** — the foundation this project builds on: the hybrid INT4+FP8 checkpoint, the INT8 lm-head, MTP-2, and the end-to-end benchmark methodology, forward-ported and recomposed here onto vLLM 0.23 + DFlash.
- [`z-lab`](https://huggingface.co/z-lab) / [Modal](https://modal.com/blog/spec-is-all-u-need) — the DFlash drafter and the block-diffusion speculative-decode work.
- [`Intel/AutoRound`](https://huggingface.co/Intel) — the INT4 target quantization.
- [`vLLM`](https://github.com/vllm-project/vllm) and the AEON sm121 image maintainers — the engine and the DFlash-enabled GB10 build.
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) — the vLLM-on-Spark base much of this ecosystem started from.

## License

MIT — see [LICENSE](LICENSE). The patches are original; vendored third-party files
(`tools/build-hybrid-checkpoint.py`) retain their upstream attribution.
