#!/usr/bin/env python3
"""v2 harness: proper bf16 GEMV baseline + a single BATCHED int8 w8a16 GEMM kernel
(one launch for any B, dot-based, pads B to >=16) vs albond's per-row loop, at the
real lm-head shape and B in {1,5,13} (base + DFlash verify-batch sizes).
"""
import time

import torch
import triton
import triton.language as tl

V, H = 248320, 3072
DEV = "cuda"
torch.manual_seed(0)

W_bf16 = (torch.randn(V, H, device=DEV, dtype=torch.float32) * 0.02).to(torch.bfloat16)
scales = (W_bf16.float().abs().amax(dim=1) / 127.0).clamp(min=1e-12)
W_int8 = (W_bf16.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
scales_f16 = scales.to(torch.float16)
INT8_BYTES = V * H
BF16_BYTES = V * H * 2


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ---- batched int8 w8a16 GEMM: out[B,N] = (x[B,K] @ (W_int8[N,K]*s[N]).T) ----
@triton.jit
def _k_batched(x_ptr, w_ptr, s_ptr, o_ptr, B, N, K,
               sxb, sxk, swn, swk, sob, son,
               BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_b = tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_b[:, None] * sxb + offs_k[None, :] * sxk
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        km = (offs_k[None, :] + k) < K
        x = tl.load(x_ptrs, mask=(offs_b[:, None] < B) & km, other=0.0).to(tl.float16)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & km, other=0).to(tl.float16)
        acc += tl.dot(x, w.T)  # [BB,BK] @ [BK,BN] -> [BB,BN]
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc * s[None, :]
    o_ptrs = o_ptr + offs_b[:, None] * sob + offs_n[None, :] * son
    tl.store(o_ptrs, acc.to(tl.float16), mask=(offs_b[:, None] < B) & (offs_n[None, :] < N))


def run_batched(x, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3):
    B = x.shape[0]
    BLOCK_B = max(16, triton.next_power_of_2(B))
    out = torch.empty(B, V, dtype=torch.float16, device=DEV)
    xf = x.to(torch.float16)
    grid = ((V + BLOCK_N - 1) // BLOCK_N,)
    _k_batched[grid](xf, W_int8, scales_f16, out, B, V, H,
                     xf.stride(0), xf.stride(1), W_int8.stride(0), W_int8.stride(1),
                     out.stride(0), out.stride(1),
                     BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                     num_warps=num_warps, num_stages=num_stages)
    return out


for B in (1, 5, 13):
    print(f"\n=== B={B} ===")
    x = torch.randn(B, H, device=DEV, dtype=torch.bfloat16) * 0.1
    ref_bf16 = (x.float() @ W_bf16.float().T)
    ref_deq = (x.float() @ (W_int8.float() * scales.unsqueeze(1)).T)
    am_floor = (ref_bf16.argmax(-1) == ref_deq.argmax(-1)).float().mean().item()
    print(f"  quant floor: argmax_match={am_floor*100:.1f}%  maxerr={(ref_deq-ref_bf16).abs().max():.4f}")

    # real bf16 GEMV baseline (stays bf16, reads 1.5GB)
    dt = bench(lambda: torch.matmul(x, W_bf16.t()))
    print(f"  {'bf16 GEMV (real baseline)':30s} {'':33s}{dt*1e3:7.3f} ms  {BF16_BYTES/dt/1e9:6.1f} GB/s")

    # batched int8 kernel, a few configs
    for (bn, bk, nw, ns) in [(128, 64, 4, 3), (256, 64, 8, 3), (128, 128, 4, 3), (64, 128, 4, 3)]:
        try:
            out = run_batched(x, bn, bk, nw, ns)
            err = (out.float() - ref_deq.float()).abs().max().item()
            am = (out.float().argmax(-1) == ref_bf16.argmax(-1)).float().mean().item()
            dt = bench(lambda: run_batched(x, bn, bk, nw, ns))
            print(f"  batched N{bn}/K{bk}/w{nw}/s{ns:<2d} maxerr={err:8.4f} argmax={am*100:5.1f}% "
                  f"{dt*1e3:7.3f} ms  {INT8_BYTES/dt/1e9:6.1f} GB/s")
        except Exception as e:
            print(f"  batched N{bn}/K{bk}/w{nw}/s{ns}: ERR {str(e)[:70]}")
