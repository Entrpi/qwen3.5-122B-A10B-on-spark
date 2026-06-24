#!/bin/bash
# Run the 4-workload acceptance/throughput bank on whichever server is up.
# $1 = container name (for the authoritative server-log accept_len). Uses the SAME
# prompts as the DFlash run so DFlash vs MTP is apples-to-apples.
CN="${1:-vllm-mtp}"
BU=http://127.0.0.1:8000; M=qwen
acc_tail(){ docker logs "$CN" 2>&1 | grep "Mean acceptance length" | tail -1 \
  | sed -E "s/.*Mean acceptance length: ([0-9.]+).*Per-position acceptance rate: ([0-9., ]+), Avg.*/accept_len=\1 per-pos=[\2]/"; }
run(){ L="$1"; P="$2"; echo "########## $L ##########"
  python3 ~/bench_decode.py --base-url $BU --model $M --max-tokens 256 --runs 4 --warmup 1 --label "$L" --prompt "$P" 2>&1 | grep -E "decode tok/s"
  echo "  server $(acc_tail)"; }
run "PROSE" "You are a careful writer. Write a long flowing essay about the history and philosophy of science, with no lists. Begin: "
run "CODE" "Write a complete Python implementation of a binary search tree class with insert, search, delete, and inorder traversal. Include docstrings and type hints. Begin:\n\nclass BSTNode:"
run "COUNTING" "Write all the whole numbers from 1 to 400, separated by commas, with no other text. Begin: 1, 2, 3, "
echo "########## HERMES (real agent turns) ##########"
python3 ~/hermes_bench.py --base-url $BU --model $M --label "HERMES" --n-samples 10 --max-tokens 200 2>&1 | grep -E "decode tok/s|accept length"
