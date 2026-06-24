#!/usr/bin/env python3
"""Validate patch_int8_lmhead_v3 landed + the helper runs on GPU. Run AFTER the
patch in the same container."""
import torch

import vllm.model_executor.layers.logits_processor as lp

print("import OK | helpers:",
      hasattr(lp, "_spark_int8_gemm"),
      hasattr(lp, "_spark_int8_lmhead_apply"),
      hasattr(lp, "_spark_k_int8"))
src = open(lp.__file__).read()
print("sentinel in _get_logits:", "DGX_SPARK_INT8_LMHEAD_V3: int8 w8a16" in src)

V, H = 4096, 512
torch.manual_seed(0)
W = torch.randn(V, H, device="cuda") * 0.02
s = (W.abs().amax(1) / 127).clamp(min=1e-12)
wi = (W / s.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8).contiguous()
sf = s.to(torch.float16)
for B in (1, 5, 13):
    x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16) * 0.1
    out = lp._spark_int8_gemm(x, wi, sf)
    ref = x.float() @ (wi.float() * s.unsqueeze(1)).T
    am = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"  B={B}: out{tuple(out.shape)} {out.dtype}  argmax={am*100:.0f}%  "
          f"maxerr={(out - ref).abs().max().item():.4f}")

# Exercise the FULL apply path (print + quantize-once + gemm) with a mock lm_head
# (vocab > 100k to trigger the int8 path). self is unused -> None.
print("--- full _spark_int8_lmhead_apply path (mock lm_head, V=131072) ---")
Vbig = 131072

class _MockLMHead:
    pass

mh = _MockLMHead()
mh.weight = (torch.randn(Vbig, H, device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16)
hs = torch.randn(3, H, device="cuda", dtype=torch.bfloat16) * 0.1
o1 = lp._spark_int8_lmhead_apply(None, mh, hs, None)  # first call: quantizes + prints
o2 = lp._spark_int8_lmhead_apply(None, mh, hs, None)  # second call: reuses int8
refb = hs.float() @ mh.weight.float().T
am = (o2.argmax(-1) == refb.argmax(-1)).float().mean().item()
print(f"  apply: out{tuple(o2.shape)} {o2.dtype}  argmax_vs_bf16={am*100:.0f}%  "
      f"int8_ready={getattr(mh, '_spark_int8_ready', None)}  weight_kept={mh.weight.numel() > 0}")
