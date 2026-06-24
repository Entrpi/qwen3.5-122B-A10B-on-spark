#!/usr/bin/env python3
"""INT8 W8A16 lm-head v3 for AEON vLLM 0.23 (sm121). Replaces ONLY the
`lm_head.quant_method.apply(...)` call inside LogitsProcessor._get_logits with a
batched int8 GEMV (one kernel launch for any batch), leaving the existing TP
gather + org_vocab_size trim untouched.

Fixes vs the broken v2 port:
  * KEEPS the bf16 lm_head weight (DFlash drafter shares it) — does NOT zero it.
    Trades the memory saving for correctness; the speed win is the int8 read in
    _get_logits, independent of keeping bf16 around.
  * Single BATCHED kernel (dot-based, pad B->16) for ALL B — no per-row Python
    loop (the v2 B>4 loop was what made spec decode SLOWER).
  * Fixed proven config (N128/K128/w4/s3, ~227 GB/s, argmax-exact vs bf16 on the
    real [248320,3072] shape) — no autotune (avoids sm121 bad-config miscompiles).

Sentinel DGX_SPARK_INT8_LMHEAD_V3. Verified standalone: int8 3.35ms vs bf16 8.8ms
(B=1) / 6.5ms (B=13); maxerr 3e-4 < quant floor, argmax 100%.
"""
import os
import sys

TARGET = "/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py"

ANCHOR = (
    "        # Get the logits for the next tokens.\n"
    "        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)\n"
)
REPLACE = (
    "        # DGX_SPARK_INT8_LMHEAD_V3: int8 w8a16 GEMV for the huge vocab projection\n"
    "        logits = _spark_int8_lmhead_apply(self, lm_head, hidden_states, embedding_bias)\n"
)

MODULE_CODE = '''

# ===================== DGX_SPARK_INT8_LMHEAD_V3 =====================
import triton as _spark_triton
import triton.language as _spark_tl


@_spark_triton.jit
def _spark_k_int8(x_ptr, w_ptr, s_ptr, o_ptr, B, N, K,
                  sxb, sxk, swn, swk, sob, son,
                  BLOCK_B: _spark_tl.constexpr, BLOCK_N: _spark_tl.constexpr,
                  BLOCK_K: _spark_tl.constexpr):
    pid_n = _spark_tl.program_id(0)
    offs_b = _spark_tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + _spark_tl.arange(0, BLOCK_N)
    offs_k = _spark_tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_b[:, None] * sxb + offs_k[None, :] * sxk
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk
    acc = _spark_tl.zeros((BLOCK_B, BLOCK_N), dtype=_spark_tl.float32)
    for k in range(0, K, BLOCK_K):
        km = (offs_k[None, :] + k) < K
        x = _spark_tl.load(x_ptrs, mask=(offs_b[:, None] < B) & km, other=0.0).to(_spark_tl.float16)
        w = _spark_tl.load(w_ptrs, mask=(offs_n[:, None] < N) & km, other=0).to(_spark_tl.float16)
        acc += _spark_tl.dot(x, w.T)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk
    s = _spark_tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(_spark_tl.float32)
    acc = acc * s[None, :]
    o_ptrs = o_ptr + offs_b[:, None] * sob + offs_n[None, :] * son
    # bf16 output matches the stock bf16 lm-head's logits dtype (F.linear keeps bf16);
    # fp32 here would ~2x the logits buffer and inflate vLLM's profiled activation reserve.
    _spark_tl.store(o_ptrs, acc.to(_spark_tl.bfloat16), mask=(offs_b[:, None] < B) & (offs_n[None, :] < N))


def _spark_int8_gemm(hidden, w_int8, w_scale):
    import torch
    N, K = w_int8.shape
    x = hidden.reshape(-1, K)
    B = x.shape[0]
    BLOCK_B = max(16, _spark_triton.next_power_of_2(B))
    out = torch.empty(B, N, dtype=torch.bfloat16, device=x.device)  # match stock bf16 logits
    xf = x.to(torch.float16)
    grid = ((N + 127) // 128,)
    _spark_k_int8[grid](xf, w_int8, w_scale, out, B, N, K,
                        xf.stride(0), xf.stride(1), w_int8.stride(0), w_int8.stride(1),
                        out.stride(0), out.stride(1),
                        BLOCK_B=BLOCK_B, BLOCK_N=128, BLOCK_K=128,
                        num_warps=4, num_stages=3)
    return out.reshape(hidden.shape[:-1] + (N,))


def _spark_int8_lmhead_apply(self, lm_head, hidden_states, embedding_bias):
    import sys
    import torch
    if not getattr(lm_head, "_spark_int8_ready", None) is True and \\
       not getattr(lm_head, "_spark_int8_disabled", False):
        w = getattr(lm_head, "weight", None)
        if (w is not None and w.dtype in (torch.bfloat16, torch.float16)
                and w.dim() == 2 and w.shape[0] > 100000):
            with torch.no_grad():
                scales = (w.float().abs().amax(dim=1) / 127.0).clamp(min=1e-12)
                w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
            lm_head._spark_w_int8 = w_int8.contiguous()
            lm_head._spark_w_scale = scales.to(torch.float16)
            lm_head._spark_int8_ready = True
            print("DGX_SPARK_INT8_LMHEAD_V3: lm_head -> int8 (%s), bf16 kept for shared drafter"
                  % (list(w_int8.shape),), file=sys.stderr, flush=True)
        else:
            lm_head._spark_int8_disabled = True
    if getattr(lm_head, "_spark_int8_ready", False) and embedding_bias is None:
        return _spark_int8_gemm(hidden_states, lm_head._spark_w_int8, lm_head._spark_w_scale)
    return lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
# =================== end DGX_SPARK_INT8_LMHEAD_V3 ===================
'''


def main():
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found"); sys.exit(1)
    src = open(TARGET).read()
    if "DGX_SPARK_INT8_LMHEAD_V3" in src:
        print("SKIP: int8 lm-head v3 already applied"); return
    if ANCHOR not in src:
        print("FAIL: _get_logits apply-anchor not found"); sys.exit(1)
    if src.count(ANCHOR) != 1:
        print(f"FAIL: anchor count={src.count(ANCHOR)} (expected 1)"); sys.exit(1)
    src = src.replace(ANCHOR, REPLACE)
    src = src + MODULE_CODE
    open(TARGET, "w").write(src)
    print("OK: int8 lm-head v3 applied")


if __name__ == "__main__":
    main()
