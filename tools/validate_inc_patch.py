#!/usr/bin/env python3
"""Validate patch_inc_hybrid.py landed on AEON 0.23's inc.py. Run AFTER the patch
in the same container: python3 /host/patch_inc_hybrid.py && python3 /host/validate_inc_patch.py
"""
import inspect

import vllm.model_executor.layers.quantization.inc as m

C = m.INCConfig
src = inspect.getsource(m)
print("import OK")
print("sentinel in source        :", "spark-dflash-hybrid-fp8" in src)
print("maybe_update_config OWN    :", "maybe_update_config" in C.__dict__)
print("_is_layer_fp8 OWN          :", "_is_layer_fp8" in C.__dict__)
print("maybe_update_config hasattr:", hasattr(C, "maybe_update_config"))
print("_is_layer_fp8 hasattr      :", hasattr(C, "_is_layer_fp8"))
# signature must accept hf_config kw (config/vllm.py calls it that way)
try:
    sig = inspect.signature(C.maybe_update_config)
    print("maybe_update_config sig    :", str(sig))
    print("accepts hf_config kw       :", "hf_config" in sig.parameters)
except Exception as e:
    print("sig err:", e)
# count FP8 dispatch sites
print("Fp8LinearMethod dispatch ct:", src.count("return Fp8LinearMethod(self.fp8_config)"))
# byte-compile sanity already implied by import; show line count
print("inc.py lines               :", len(src.splitlines()))
