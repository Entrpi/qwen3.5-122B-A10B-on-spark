#!/usr/bin/env python3
"""spark-dflash-hybrid-fp8: port of albond's hybrid INT4+FP8 dispatch
(patches/01-hybrid-int4-fp8/inc.py.patch) onto AEON 0.23's vllm INCConfig.

Adds, to vllm/model_executor/layers/quantization/inc.py:
  * INCConfig.fp8_config / fp8_layers fields
  * maybe_update_config OVERRIDE — AEON 0.23 signature (model_name, hf_config=None,
    revision=None); the base hook is already CALLED from config/vllm.py:634, so we
    only supply the override. Scans the checkpoint's safetensors metadata for
    float8_e4m3fn weights that have a .weight_scale_inv, builds an Fp8Config, and
    records those layer prefixes.
  * _is_layer_fp8 — exact + fused + substring match against fp8_layers
  * FP8 dispatch at BOTH dense short-circuits: get_quant_method's extra_config
    override AND the apply_*_quant_layer not-quantized blocks.

Idempotent; sentinel 'spark-dflash-hybrid-fp8'. Mirrors patch_unify2.py's style.
"""
import sys

import vllm.model_executor.layers.quantization.inc as inc_mod

path = inc_mod.__file__
src = open(path).read()
SENT = "spark-dflash-hybrid-fp8"
if SENT in src:
    print(f"[patch_inc_hybrid] already applied: {path}")
    sys.exit(0)

# 1. __init__ fields (anchor unique: only INCConfig has pack_factor = Fraction(32,..))
a1 = "        self.pack_factor = Fraction(32, weight_bits)\n"
b1 = a1 + (
    "        # spark-dflash-hybrid-fp8: populated by maybe_update_config\n"
    "        self.fp8_config = None\n"
    "        self.fp8_layers = set()\n"
)
assert src.count(a1) == 1, f"anchor1 count={src.count(a1)}"
src = src.replace(a1, b1)

# 2. apply_vllm_mapper: remap fp8_layers (HF->vLLM names) after extra_config remap
a2 = (
    "        if self.extra_config is not None:\n"
    "            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)\n"
)
b2 = a2 + (
    "        if self.fp8_layers:  # spark-dflash-hybrid-fp8\n"
    "            self.fp8_layers = set(\n"
    "                hf_to_vllm_mapper.apply_list(list(self.fp8_layers))\n"
    "            )\n"
)
assert src.count(a2) == 1, f"anchor2 count={src.count(a2)}"
src = src.replace(a2, b2)

# 3. insert maybe_update_config + _is_layer_fp8 before apply_awq_quant_layer
a3 = '    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):\n'
methods = '''    def maybe_update_config(  # spark-dflash-hybrid-fp8
        self,
        model_name: str,
        hf_config=None,
        revision: str | None = None,
    ):
        """Detect FP8 dense layers in a hybrid INT4+FP8 checkpoint."""
        import torch as _torch
        from safetensors.torch import _TYPES as _SF
        from vllm.transformers_utils.config import get_safetensors_params_metadata
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        fp8_weights = {}
        for pn, info in metadata.items():
            ds = info.get("dtype", None)
            if ds is None:
                continue
            if _SF.get(ds) == _torch.float8_e4m3fn and pn.endswith(".weight"):
                sn = pn.replace(".weight", ".weight_scale_inv")
                if sn in metadata:
                    fp8_weights[pn] = info
        if not fp8_weights:
            logger.info("spark-dflash-hybrid-fp8: no FP8 dense layers detected")
            return
        block_size = None
        for pn, info in fp8_weights.items():
            sn = pn.replace(".weight", ".weight_scale_inv")
            ws = info.get("shape", [])
            ss = metadata[sn].get("shape", [])
            if len(ws) == 2 and len(ss) == 2:
                block_size = [ws[0] // ss[0], ws[1] // ss[1]]
                break
        if block_size is None:
            return
        self.fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=block_size,
        )
        self.fp8_layers = {n.rsplit(".weight", 1)[0] for n in fp8_weights}
        _sample = sorted(self.fp8_layers)[:3]
        logger.info(
            "spark-dflash-hybrid-fp8: detected %d FP8 dense layers "
            "(block_size=%s) e.g. %s",
            len(self.fp8_layers), block_size, _sample,
        )

    def _is_layer_fp8(self, prefix: str) -> bool:  # spark-dflash-hybrid-fp8
        if not self.fp8_layers:
            return False
        if prefix in self.fp8_layers:
            return True
        fused = getattr(self, "packed_modules_mapping", {})
        proj = prefix.split(".")[-1]
        if proj in fused:
            shards = [prefix.replace(proj, s) for s in fused[proj]]
            return all(
                any(fl in sp for fl in self.fp8_layers) for sp in shards
            )
        return any(fl in prefix for fl in self.fp8_layers)

'''
assert src.count(a3) == 1, f"anchor3 count={src.count(a3)}"
src = src.replace(a3, methods + a3)

# 4. FP8 dispatch in the not-quantized blocks (awq/gptq/xpu/cpu are byte-identical;
#    guard is inert unless fp8_config set, so patching all is safe)
a4 = (
    "        if not self.check_quantized(weight_bits):\n"
    "            if isinstance(layer, (LinearBase, ParallelLMHead)):\n"
    "                return UnquantizedLinearMethod()\n"
    "            else:\n"
    "                return None\n"
)
b4 = (
    "        if not self.check_quantized(weight_bits):\n"
    "            if self.fp8_config and self._is_layer_fp8(prefix):  # spark-dflash-hybrid-fp8\n"
    "                from vllm.model_executor.layers.quantization.fp8 import (\n"
    "                    Fp8LinearMethod,\n"
    "                )\n"
    "                return Fp8LinearMethod(self.fp8_config)\n"
    "            if isinstance(layer, (LinearBase, ParallelLMHead)):\n"
    "                return UnquantizedLinearMethod()\n"
    "            else:\n"
    "                return None\n"
)
n4 = src.count(a4)
assert n4 >= 2, f"anchor4 count={n4}"
src = src.replace(a4, b4)

# 5. FP8 dispatch in get_quant_method's extra_config (bits>=16) override
a5 = (
    '                ) and self.extra_config[layer_name].get("bits", 16) >= 16:\n'
    "                    return UnquantizedLinearMethod()\n"
)
b5 = (
    '                ) and self.extra_config[layer_name].get("bits", 16) >= 16:\n'
    "                    if self.fp8_config and self._is_layer_fp8(prefix):  # spark-dflash-hybrid-fp8\n"
    "                        from vllm.model_executor.layers.quantization.fp8 import (\n"
    "                            Fp8LinearMethod,\n"
    "                        )\n"
    "                        return Fp8LinearMethod(self.fp8_config)\n"
    "                    return UnquantizedLinearMethod()\n"
)
assert src.count(a5) == 1, f"anchor5 count={src.count(a5)}"
src = src.replace(a5, b5)

open(path, "w").write(src)
print(f"[patch_inc_hybrid] applied {SENT} to {path} (not-quant blocks x{n4})")
