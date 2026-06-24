#!/bin/bash
# mtp_serve.sh — native qwen3_5 MTP-N head for the comparison (`--profile mtp`).
# The MTP head (1 layer, reuses target KV/embed/lm_head) is in the Intel
# checkpoint (mtp.layers.0) — no separate drafter, no unify patch needed.
#   $1 = num_speculative_tokens (default 2 = the "MTP-2" recipe); $2 = backend.
set -euo pipefail
NSPEC="${1:-2}"
BACKEND="${2:-flash_attn}"
MODEL="${MODEL:-Intel/Qwen3.5-122B-A10B-int4-AutoRound}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM="${GPU_MEM:-0.8}"
PORT="${PORT:-8000}"
echo "[mtp] qwen3_5_mtp — backend=$BACKEND, num_speculative_tokens=$NSPEC, model=$MODEL"
exec vllm serve "$MODEL" \
  --served-model-name qwen \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 16 \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code \
  --attention-backend "$BACKEND" \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$NSPEC,\"model\":\"$MODEL\"}"
