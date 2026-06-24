#!/usr/bin/env bash
# Monitor SGLang container startup with an OOM auto-kill guard.
# Breaks on: READY, ERROR (traceback), OOM-GUARD (avail mem too low), or timeout.
set -u
NAME="${1:-sglang-dflash}"
MAX_ITERS="${2:-60}"     # 60 * 20s = 20 min
FLOOR_MB="${3:-4096}"    # kill if available memory drops below this

for i in $(seq 1 "$MAX_ITERS"); do
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    echo "STATE=EXITED iter=$i"
    echo "--- last log ---"; docker logs "$NAME" 2>&1 | tail -25
    exit 0
  fi
  AVAIL=$(free -m | awk '/^Mem:/{print $7}')
  if [ "$AVAIL" -lt "$FLOOR_MB" ]; then
    echo "STATE=OOM-GUARD iter=$i avail_mb=$AVAIL  -> killing $NAME"
    docker kill "$NAME" >/dev/null 2>&1
    echo "--- last log ---"; docker logs "$NAME" 2>&1 | tail -25
    exit 0
  fi
  LOG=$(docker logs "$NAME" 2>&1)
  if echo "$LOG" | grep -qiE "server is fired up|Application startup complete|The server is ready"; then
    echo "STATE=READY iter=$i avail_mb=$AVAIL"
    echo "--- tail ---"; echo "$LOG" | tail -20
    exit 0
  fi
  if echo "$LOG" | grep -qE "Traceback \(most recent call last\)|CUDA out of memory|RuntimeError|AssertionError|ValueError|raise NotImplementedError"; then
    echo "STATE=ERROR iter=$i avail_mb=$AVAIL"
    echo "--- tail ---"; echo "$LOG" | tail -40
    exit 0
  fi
  LAST=$(echo "$LOG" | tail -1 | cut -c1-110)
  echo "iter=$i avail_mb=$AVAIL :: $LAST"
  sleep 20
done
echo "STATE=TIMEOUT after $MAX_ITERS iters"
docker logs "$NAME" 2>&1 | tail -20
