"""FSDP2: parameters, gradients and optimizer state are sharded across ranks.

Each block is its own FSDP unit (fully_shard), plus the root for the embedding / lm_head /
final norm. Before a unit's forward its parameters are all-gathered; after it they are freed
(reshard_after_forward) and gathered again for the backward, whose gradients are
reduce-scattered so each rank keeps the gradient of its own shard.

Explicit prefetch issues the next block's all-gather (the previous block's, in backward)
before the current block computes, instead of relying on FSDP2's default of one-ahead
implicit prefetch, so communication has more compute to hide behind.
"""

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

PARAM_DTYPES = {"fp32": None, "fp16": torch.float16}


def shard(model: nn.Module, mesh: DeviceMesh, param_dtype: str, reshard_after_forward: bool, prefetch: bool,
          compile_blocks: bool) -> nn.Module:
    # param_dtype fp16 gathers half-size parameters (half the traffic); gradients are still
    # reduced in FP32 so the sum over ranks doesn't lose precision.
    mp = MixedPrecisionPolicy(param_dtype=PARAM_DTYPES[param_dtype], reduce_dtype=torch.float32)
    blocks = list(model.blocks)
    for block in blocks:
        if compile_blocks:
            block.compile()  # per-block compile: one graph per FSDP unit, collectives stay outside
        fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    if prefetch:
        for i, block in enumerate(blocks):
            if i + 1 < len(blocks):
                block.set_modules_to_forward_prefetch([blocks[i + 1]])
            if i > 0:
                block.set_modules_to_backward_prefetch([blocks[i - 1]])
    return model
