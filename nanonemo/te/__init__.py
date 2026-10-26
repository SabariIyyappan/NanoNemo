"""Transformer Engine variant of the model (M8, AWS L4): the blocks' seven linear layers become
te.Linear, which can run its matmuls in FP8 inside te.fp8_autocast. Everything else (norms,
RoPE, attention, the tied LM head) is unchanged, so the FP8 row differs from BF16 only in
those GEMMs.

FP8 needs Ada (sm_89) or newer; on Turing te.Linear still runs in FP16/FP32, which is how
the layer swap itself is tested on the laptop.

Recipes:
  delayed  DelayedScaling: each tensor's scale comes from the max |value| seen over the
           last steps (amax history), so no extra pass over the data per step.
  current  Float8CurrentScaling: the scale comes from this step's own max |value|, one more
           read of each tensor, but no stale scales.
FP8 format HYBRID: E4M3 (more mantissa) for the forward, E5M2 (more range) for gradients.
"""

import torch


def linear(in_features: int, out_features: int):
    import transformer_engine.pytorch as te

    layer = te.Linear(in_features, out_features, bias=False, params_dtype=torch.float32)
    layer._nanonemo_init = True  # so NanoNemo's init applies the same N(0, 0.02) as nn.Linear
    return layer


def fp8_context(recipe: str):
    """A factory for the forward's FP8 context (call it to get a fresh context manager)."""
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format

    ok = te.is_fp8_available()
    ok, why = ok if isinstance(ok, tuple) else (ok, "")
    if not ok:
        raise RuntimeError(f"FP8 not available on this GPU (needs sm_89+): {why}")
    r = {"delayed": lambda: DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"),
         "current": lambda: Float8CurrentScaling(fp8_format=Format.HYBRID)}[recipe]()
    return lambda: te.fp8_autocast(enabled=True, fp8_recipe=r)


