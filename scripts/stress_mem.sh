#!/bin/bash
# stress_mem.sh — drive the server with the concurrency bank while sampling host
# memory, to prove a gpu-memory-utilization setting survives peak load WITHOUT
# swapping (the failure mode that hard-freezes the Spark). Reports min available
# RAM and max swap observed across the run.
#
#   usage: stress_mem.sh [LEVELS]   (default "1,2,3")
# Run on the box; the server (container qwen-spark) must be READY on :8000.
set -uo pipefail
LEVELS="${1:-1,2,3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=/tmp/stress_memlog.$$

# baseline
echo "[stress] swap before:"; free -m | awk '/Swap:/{print "  swap_used_MiB",$3}'
echo "[stress] avail before:"; free -m | awk '/Mem:/{print "  avail_MiB",$7}'

# background sampler: timestamp, avail MiB, swap-used MiB, every 1s
( for i in $(seq 1 900); do
    free -m | awk -v t="$i" '/Mem:/{a=$7} /Swap:/{s=$3} END{print t, a, s}'
    sleep 1
  done ) > "$LOG" 2>/dev/null &
SPID=$!

echo "[stress] running conc_workloads --levels $LEVELS ..."
python3 "$HERE/conc_workloads.py" --levels "$LEVELS"
RC=$?

kill "$SPID" 2>/dev/null
echo "[stress] === memory envelope during load ==="
awk 'NF>=3 {if(min==""||$2<min)min=$2; if($3>maxs)maxs=$3} END{print "  min_avail_MiB", min, " max_swap_MiB", maxs+0}' "$LOG"
rm -f "$LOG"
echo "[stress] conc_workloads exit=$RC"
