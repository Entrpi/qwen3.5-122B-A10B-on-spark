#!/usr/bin/env bash
# install.sh — Qwen3.5-122B-A10B + DFlash speculative decode on NVIDIA DGX Spark
#              (GB10 / SM121, 128 GB / 119 GiB unified), via vLLM in Docker.
#
#   curl -sSL https://raw.githubusercontent.com/Entrpi/qwen3.5-122B-A10B-on-spark/main/install.sh | bash
#   curl -sSL https://raw.githubusercontent.com/Entrpi/qwen3.5-122B-A10B-on-spark/main/install.sh | bash -s -- --help
#
# What this does (every step idempotent — safe to re-run):
#
#   1. Verifies the host is a DGX Spark / GB10 (SM121) with Docker + the NVIDIA
#      container runtime, and enough free disk for the chosen profile.
#   2. Pulls the prebuilt sm121 vLLM image (DFlash-enabled, vLLM 0.23).
#   3. Downloads the INT4 target + DFlash drafter from Hugging Face into the HF
#      cache  — OR reuses a checkpoint you already have (--model-dir / --hf-home).
#   4. (optional) Builds the hybrid INT4+FP8 checkpoint for the "dense" profile.
#   5. Starts the vLLM server for the chosen --profile, waits until READY, and
#      runs the "capital of France" smoke test (expects "Paris").
#
# The script makes NO changes outside:
#   - the Docker image cache              (the pulled image)
#   - $HF_HOME                            (default ~/.cache/huggingface)
#   - $HYBRID_DIR                         (only with --build-hybrid)
#   - the running container named $NAME   (only with --start / smoke)
#
# This repo provides the install + serve + patch + benchmark layer ON TOP of:
#   - Intel/Qwen3.5-122B-A10B-int4-AutoRound  (target weights)
#   - z-lab/Qwen3.5-122B-A10B-DFlash          (block-diffusion drafter)
#   - ghcr.io/aeon-7/aeon-vllm-ultimate       (sm121 DFlash-enabled vLLM)
#
# License: MIT.  Source: https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark

set -euo pipefail

# ============================================================================
# 0. defaults + flag parsing
# ============================================================================

# Prebuilt sm121 vLLM image with the DFlash PRs + the .pth that auto-applies our
# KV-unify patch is NOT baked in — we apply patches at serve time from runtime/.
IMAGE="${QWEN_IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix}"

TARGET_REPO="${TARGET_REPO:-Intel/Qwen3.5-122B-A10B-int4-AutoRound}"   # INT4 target (~62 GiB)
DRAFT_REPO="${DRAFT_REPO:-z-lab/Qwen3.5-122B-A10B-DFlash}"             # 0.8B drafter (~1.6 GiB)
FP8_REPO="${FP8_REPO:-Qwen/Qwen3.5-122B-A10B-FP8}"                     # FP8 donor for --build-hybrid

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HYBRID_DIR="${HYBRID_DIR:-$HOME/qwen3.5-122b-hybrid-int4-fp8}"
MODEL_DIR=""                       # --model-dir: a pre-downloaded INT4 checkpoint dir

# This repo's own dir (works for `curl | bash` too: falls back to a clone).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "$HOME/code/qwen3.5-122B-A10B-on-spark")}"
REPO_URL="${REPO_URL:-https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark.git}"

NAME="${NAME:-qwen-spark}"
PROFILE="dflash"                   # dflash | dense | base | mtp
NSPEC=""                           # override num_speculative_tokens (default per profile)
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"               # max-model-len: model native max (KV is ~24 KiB/token)
GPU_MEM="${GPU_MEM:-0.82}"         # VALIDATED: ~14 GiB free (0.88+ over-subscribes -> swap)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"          # 3 concurrent streams (e.g. 3 subagents); pool ~457k tok, 1.74x @ 262k
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"  # chunked-prefill chunk (decoupled from ctx)
BACKEND="${BACKEND:-flash_attn}"

FORCE_HW=0
SKIP_PULL=0
SKIP_DOWNLOAD=0
BUILD_HYBRID=0
START_SERVER=0
SKIP_SMOKE=0

usage() {
    cat <<EOF
Usage: $0 [flags]

Profiles (--profile):
  dflash   INT4 target + DFlash drafter, n=12      (DEFAULT — best for agents/Hermes;
                                                    ~81 tok/s on real tool-call turns)
  dense    hybrid INT4+FP8 + int8 lm-head + DFlash  (the dense-bandwidth stack;
                                                    +28% at base, +10% low-accept spec.
                                                    Needs --build-hybrid first.)
  base     plain INT4, no speculative decode        (~28 tok/s c=1 baseline)
  mtp      INT4 + native MTP-2 head                  (the albond comparison path)

Flags:
  --help                  Show this help.
  --profile NAME          One of dflash|dense|base|mtp (default: dflash).
  --start                 Start the vLLM server + smoke test after setup.
  --build-hybrid          Build the hybrid INT4+FP8 checkpoint (~20 min, needs FP8 donor).
  --no-pull               Skip docker pull (use the local image).
  --no-download           Skip HF download (assume target+drafter already cached).
  --model-dir DIR         Use a pre-downloaded INT4 target checkpoint dir (skip its
                          download; mounted read-only at /model).
  --hf-home DIR           Use/populate this HF cache dir (default: $HF_HOME).
  --nspec N               num_speculative_tokens (default 12 dflash/dense, 2 mtp, 0 base).
  --port N                Server port (default: $PORT).
  --ctx N                 max-model-len (default: $CTX = model native max).
  --gpu-mem F             gpu-memory-utilization (default: $GPU_MEM = ~14 GiB free; 0.88+ over-subscribes/swaps).
  --max-num-seqs N        concurrent sequences (default: $MAX_NUM_SEQS; KV pool ~457k tokens, 1.74x at full 262k).
  --max-batched-tokens N  chunked-prefill chunk (default: $MAX_BATCHED_TOKENS; keep < ctx).
  --force                 Skip the GB10/SM121 host check.
  --no-smoke              Start the server but skip the Paris smoke test.

Environment equivalents:
  QWEN_IMAGE TARGET_REPO DRAFT_REPO FP8_REPO HF_HOME HYBRID_DIR
  NAME PORT CTX GPU_MEM MAX_NUM_SEQS MAX_BATCHED_TOKENS BACKEND REPO_DIR REPO_URL
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --start) START_SERVER=1; shift ;;
        --build-hybrid) BUILD_HYBRID=1; shift ;;
        --no-pull) SKIP_PULL=1; shift ;;
        --no-download) SKIP_DOWNLOAD=1; shift ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --hf-home) HF_HOME="$2"; shift 2 ;;
        --nspec) NSPEC="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --max-batched-tokens) MAX_BATCHED_TOKENS="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --force) FORCE_HW=1; shift ;;
        --no-smoke) SKIP_SMOKE=1; shift ;;
        *) echo "Unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

case "$PROFILE" in dflash|dense|base|mtp) ;; *) echo "Bad --profile: $PROFILE" >&2; exit 2 ;; esac

c_red()   { printf '\033[31m%s\033[0m' "$*"; }
c_green() { printf '\033[32m%s\033[0m' "$*"; }
c_yellow(){ printf '\033[33m%s\033[0m' "$*"; }
log() { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*"; }
die() { printf '\n%s %s\n' "$(c_red FATAL:)" "$*" >&2; exit 1; }
warn(){ printf '%s %s\n' "$(c_yellow WARN:)" "$*" >&2; }
ok()  { printf '%s %s\n' "$(c_green OK:)" "$*"; }

# ============================================================================
# 1. host verification
# ============================================================================

verify_host() {
    log "Verifying host..."
    local m; m=$(uname -m)
    if [[ "$m" != "aarch64" ]] && [[ "$FORCE_HW" -eq 0 ]]; then
        die "Expected aarch64 (Grace+Blackwell); got $m. Pass --force to skip."
    fi
    command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker + the NVIDIA container runtime."
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Need the NVIDIA driver."
    local gpu; gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)
    [[ -n "$gpu" ]] || die "nvidia-smi failed to enumerate GPUs."
    log "GPU: $gpu"
    if ! echo "$gpu" | grep -qE '12\.1|GB10|Spark'; then
        [[ "$FORCE_HW" -eq 1 ]] || die "Not detecting GB10/SM12.1. Pass --force (and maybe --backend) to proceed."
        warn "Host is not GB10/SM121; proceeding under --force (untested)."
    fi
    # Docker can see the GPU? (--entrypoint true so we test GPU access, not the image's CMD)
    if ! docker info 2>/dev/null | grep -qiE 'nvidia|Default Runtime: nvidia' \
       && ! docker run --rm --gpus all --entrypoint true "$IMAGE" 2>/dev/null; then
        warn "Could not confirm Docker GPU access (nvidia-container-toolkit). 'docker run --gpus all' must work."
    fi
    # Disk
    local need=75; [[ "$BUILD_HYBRID" -eq 1 ]] && need=150
    local free; free=$(df -BG "$HOME" | awk 'NR==2{gsub("G","",$4);print $4}')
    if (( free < need )) && [[ "$SKIP_DOWNLOAD" -eq 0 ]] && [[ -z "$MODEL_DIR" ]]; then
        die "Need >= ${need} GiB free under $HOME for profile '$PROFILE'; have ${free} GiB. Use --model-dir / --no-download, or free space."
    fi
    ok "Host checks passed (free ${free} GiB)."
}

# ============================================================================
# 2. pull image
# ============================================================================

pull_image() {
    if [[ "$SKIP_PULL" -eq 1 ]]; then log "Skipping docker pull (--no-pull)."; return; fi
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then ok "Image present: $IMAGE"; return; fi
    log "Pulling $IMAGE (~40 GiB, one-time) ..."
    docker pull "$IMAGE"
    ok "Image pulled."
}

# ============================================================================
# 3. download models (idempotent — snapshot_download no-ops if cached)
# ============================================================================

hf_get() {  # repo -> populate HF cache via the image's huggingface_hub
    local repo="$1"
    docker run --rm --net=host -e HF_HOME=/hf ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
        -v "$HF_HOME:/hf" --entrypoint python3 "$IMAGE" \
        -c "from huggingface_hub import snapshot_download as s; s('$repo')"
}

download_models() {
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then log "Skipping HF download (--no-download)."; return; fi
    mkdir -p "$HF_HOME"
    if [[ -n "$MODEL_DIR" ]]; then
        [[ -f "$MODEL_DIR/config.json" ]] || die "--model-dir $MODEL_DIR has no config.json"
        log "Using pre-downloaded target at $MODEL_DIR (skipping target download)."
    else
        log "Fetching target $TARGET_REPO into $HF_HOME ..."
        hf_get "$TARGET_REPO"
    fi
    log "Fetching drafter $DRAFT_REPO ..."
    hf_get "$DRAFT_REPO"
    ok "Models ready."
}

# ============================================================================
# 4. optional: build the hybrid INT4+FP8 checkpoint
# ============================================================================

build_hybrid() {
    [[ "$BUILD_HYBRID" -eq 1 ]] || return 0
    if [[ -f "$HYBRID_DIR/model.safetensors.index.json" ]]; then ok "Hybrid ckpt present: $HYBRID_DIR"; return; fi
    local gptq="$MODEL_DIR"
    if [[ -z "$gptq" ]]; then
        gptq=$(docker run --rm -v "$HF_HOME:/hf" -e HF_HOME=/hf --entrypoint python3 "$IMAGE" \
            -c "from huggingface_hub import snapshot_download as s; print(s('$TARGET_REPO'))" | tail -1)
        gptq="/hf-snap"   # mount the cache; resolve inside the container below
    fi
    mkdir -p "$HYBRID_DIR"
    log "Building hybrid INT4+FP8 checkpoint -> $HYBRID_DIR (~20 min) ..."
    docker run --rm --net=host -e HF_HOME=/hf ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
        -v "$HF_HOME:/hf" -v "$HYBRID_DIR:/out" -v "$REPO_DIR/tools:/tools:ro" \
        ${MODEL_DIR:+-v "$MODEL_DIR:/gptq:ro"} \
        --entrypoint bash "$IMAGE" -c '
            set -e
            GPTQ="'"${MODEL_DIR:+/gptq}"'"
            if [ -z "$GPTQ" ]; then
              GPTQ=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s(\"'"$TARGET_REPO"'\"))")
            fi
            python3 /tools/build-hybrid-checkpoint.py --gptq-dir "$GPTQ" \
                --fp8-repo "'"$FP8_REPO"'" --output /out --force
            rm -rf /out/.fp8_cache'
    ok "Hybrid checkpoint built: $HYBRID_DIR"
}

# ============================================================================
# 5. start server (+ smoke test)
# ============================================================================

ensure_runtime() {  # make sure runtime/ (serve wrapper + patches) is on disk
    if [[ -f "$REPO_DIR/runtime/serve.sh" ]]; then return; fi
    log "runtime/ not found next to install.sh — cloning repo to $HOME/code/qwen3.5-122B-A10B-on-spark"
    REPO_DIR="$HOME/code/qwen3.5-122B-A10B-on-spark"
    [[ -d "$REPO_DIR/.git" ]] || git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    [[ -f "$REPO_DIR/runtime/serve.sh" ]] || die "runtime/serve.sh still missing after clone."
}

start_server() {
    [[ "$START_SERVER" -eq 1 ]] || { log "Setup complete. Re-run with --start to launch the server."; return; }
    ensure_runtime

    # profile -> serve args + env + mounts
    local nspec model_env=() mounts=() serve_args
    case "$PROFILE" in
        dflash) nspec="${NSPEC:-12}"; serve_args="$nspec $BACKEND" ;;
        dense)  nspec="${NSPEC:-12}"; serve_args="$nspec $BACKEND"
                [[ -f "$HYBRID_DIR/model.safetensors.index.json" ]] || die "dense profile needs the hybrid ckpt — run with --build-hybrid first."
                model_env=(-e MODEL=/model -e INC_HYBRID=1 -e INT8_LMHEAD_V3=1)
                mounts=(-v "$HYBRID_DIR:/model:ro") ;;
        base)   nspec="${NSPEC:-0}"; serve_args="$nspec $BACKEND" ;;
        mtp)    nspec="${NSPEC:-2}"; serve_args="$nspec $BACKEND" ;;
    esac
    if [[ -n "$MODEL_DIR" && "$PROFILE" != "dense" ]]; then
        model_env=(-e MODEL=/model); mounts=(-v "$MODEL_DIR:/model:ro")
    fi
    local wrapper="/host/serve.sh"; [[ "$PROFILE" == "mtp" ]] && wrapper="/host/mtp_serve.sh"

    log "Starting profile=$PROFILE (nspec=$nspec, ctx=$CTX, gpu-mem=$GPU_MEM) as container '$NAME' ..."
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "$NAME" --gpus all --net=host --ipc=host --ulimit memlock=-1:-1 \
        -e HF_HOME=/hf -e MAX_MODEL_LEN="$CTX" -e GPU_MEM="$GPU_MEM" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e MAX_BATCHED_TOKENS="$MAX_BATCHED_TOKENS" ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
        "${model_env[@]}" \
        -v "$HF_HOME:/hf" -v "$REPO_DIR/runtime:/host:ro" "${mounts[@]}" \
        --entrypoint bash "$IMAGE" "$wrapper" $serve_args >/dev/null
    log "Container started. Model load + compile is ~8-12 min. Tail: docker logs -f $NAME"

    log "Waiting for http://127.0.0.1:$PORT/health ..."
    local i
    for i in $(seq 1 180); do
        if ! docker ps --format '{{.Names}}' | grep -q "^$NAME$"; then
            docker logs "$NAME" 2>&1 | tail -30; die "Container exited during load. See log above."
        fi
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            ok "Server READY on http://127.0.0.1:$PORT"
            break
        fi
        sleep 5
    done
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || die "Server not ready within ~15 min. docker logs $NAME"

    [[ "$SKIP_SMOKE" -eq 1 ]] && { log "Skipping smoke test (--no-smoke)."; return; }
    log "Smoke test: 'capital of France' ..."
    local out
    out=$(curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
          -d "{\"model\":\"qwen\",\"prompt\":\"What is the capital of France? Answer in one word.\",\"max_tokens\":8,\"temperature\":0}" \
          | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || true)
    echo "  -> $out"
    echo "$out" | grep -qi paris && ok "Smoke test PASSED — 'Paris'." || die "Smoke test FAILED — 'Paris' not in output."
}

# ============================================================================
# main
# ============================================================================

verify_host
pull_image
download_models
build_hybrid
start_server

echo
ok "Done (profile=$PROFILE)."
echo "  Server:   http://127.0.0.1:$PORT/v1   (model name: qwen)"
echo "  Logs:     docker logs -f $NAME"
echo "  Stop:     docker rm -f $NAME"
echo "  Bench:    python3 scripts/bench_decode.py --base-url http://127.0.0.1:$PORT --model qwen --prompt 'Write an essay about tea.'"
echo "  Agent:    python3 scripts/hermes_bench.py --base-url http://127.0.0.1:$PORT   # real tool-call turns"
