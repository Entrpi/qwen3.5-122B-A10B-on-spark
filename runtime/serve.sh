#!/bin/bash
# serve.sh — runs INSIDE the sm121 vLLM container (mounted at /host). Applies the
# runtime monkeypatches, then `vllm serve`s the Qwen3.5-122B-A10B INT4 target with
# the DFlash drafter. Driven by install.sh; can also be run by hand.
#
#   args:  $1 = num_speculative_tokens (0 = no-spec baseline)
#          $2 = target attention backend (flash_attn | FLASHINFER)
#   env:   MODEL            target path/repo (default Intel INT4; /model for hybrid)
#          INC_HYBRID=1     apply the hybrid INT4+FP8 dense-expert dispatch patch
#          INT8_LMHEAD_V3=1 apply the int8 lm-head GEMV patch
#          MAX_MODEL_LEN GPU_MEM PORT
#
# Stack rationale: the DFlash drafter is non-causal -> needs FLASH_ATTN (FA2). The
# hybrid GDN+mamba+MoE target's KV page geometry won't absorb the drafter's
# attention spec without patch_unify2 (scale-block unify) + prefix-caching OFF
# (NoPrefixCache coordinator, dodges the hash assert). See docs/FINDINGS.md.
set -euo pipefail
NSPEC="${1:-12}"
BACKEND="${2:-flash_attn}"
MODEL="${MODEL:-Intel/Qwen3.5-122B-A10B-int4-AutoRound}"
DRAFT="${DRAFT:-z-lab/Qwen3.5-122B-A10B-DFlash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"      # model native max; KV is ~24 KiB/token so it fits
GPU_MEM="${GPU_MEM:-0.82}"                     # VALIDATED: ~14 GiB free on 128 GB (119 GiB) GB10 (0.88+ over-subscribes -> swap)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"             # 3 concurrent streams; KV pool ~427k (dense) / ~457k (dflash) tokens at full 262144
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"  # chunked-prefill chunk (NOT = max-model-len)
PORT="${PORT:-8000}"
# Read straight to the device (no mmap, no host staging) — the slow default safetensors
# read+copy is ~8 min on Spark; fastsafetensors cuts it to ~1 min. Falls back to nogds
# automatically. Override LOAD_FORMAT=auto|safetensors if the pkg is absent (or set
# --safetensors-load-strategy eager via SAFETENSORS_STRATEGY).
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"

# Reclaim vLLM's CUDA-graph memory OVER-estimate back to the KV pool. The profiler
# reserves ~0.7 GiB for the graph pool but capture actually uses ~0.14 GiB; disabling
# the estimate gives the difference (~0.6 GiB / ~9.5k tokens) to KV. The real capture
# then comes out of the (1 - gpu-mem) headroom, which is ~21 GiB at 0.82 -> no OOM
# risk at the shipped util. Set ESTIMATE_CUDAGRAPHS=1 to restore vLLM's default if you
# push gpu-mem very high (small headroom).
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"

# FLA sm121 big-tile shmem fix (prefill/TTFT only on sm121; harmless, free).
echo "[serve] FLA sm121 big-tile shmem patch"
python3 /host/patch_fla_shmem.py || true

if [ "${INC_HYBRID:-0}" = "1" ]; then
  echo "[serve] hybrid INT4+FP8 dispatch patch (inc.py)"
  python3 /host/patch_inc_hybrid.py
fi
if [ "${INT8_LMHEAD_V3:-0}" = "1" ]; then
  echo "[serve] int8 lm-head v3 patch (batched w8a16 GEMV)"
  python3 /host/patch_int8_lmhead_v3.py
fi

if [ "$NSPEC" = "0" ]; then
  SPEC_ARG=()
  echo "[serve] NO-SPEC baseline (identical flags, prefix-off)"
else
  SPEC_ARG=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NSPEC,\"attention_backend\":\"FLASH_ATTN\"}")
  echo "[serve] DFlash n=$NSPEC, target-backend=$BACKEND, drafter=FLASH_ATTN, model=$MODEL"
fi
python3 /host/patch_unify2.py || { [ "$NSPEC" = "0" ] && true; }

exec vllm serve "$MODEL" \
  --served-model-name qwen \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEM" \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code \
  --load-format "$LOAD_FORMAT" \
  --attention-backend "$BACKEND" \
  "${SPEC_ARG[@]}"
