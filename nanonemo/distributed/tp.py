"""Tensor parallel (Megatron-style) + sequence parallel, with DTensor.

Inside each block:
  attention: wq/wk/wv column-parallel (each rank computes its own heads: 12 Q -> 6, 4 KV -> 2
             at TP=2, so GQA groups stay whole on one rank), wo row-parallel.
  MLP:       gate/up column-parallel, down row-parallel.
Between those the residual stream is sharded along the sequence (sequence parallel): the norms
and residual adds run on T/tp tokens per rank. Entering attention / MLP the sequence is
all-gathered; leaving them, the row-parallel output is reduce-scattered back to sequence
shards, which replaces the all-reduce that plain TP would need.

The embedding is row-parallel over the vocabulary, and the tied lm_head reuses the same
vocab-sharded weight as a column-parallel layer: each rank produces logits for its slice of
the vocabulary, and the loss is computed on those shards under loss_parallel(), so the full
logits are never gathered.

The model's own kernels run on local (plain) tensors inside attention and the MLP, so RoPE,
SwiGLU and attention kernels work unchanged. The norms and the loss see DTensors, so they
must be the PyTorch versions.
"""

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)


def parallelize(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    cfg, tp = model.cfg, mesh.size()
    if cfg.n_heads % tp or cfg.n_kv_heads % tp:
        raise ValueError(f"TP={tp} must divide n_heads {cfg.n_heads} and n_kv_heads {cfg.n_kv_heads}")
    if cfg.norm_kernel != "torch" or cfg.loss_kernel != "torch":
        raise ValueError("TP runs the norms and the loss on DTensors: norm_kernel and loss_kernel must be 'torch'")

    parallelize_module(model, mesh, {
        "tok_emb": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=False),
        # The final norm's output is sequence-sharded; the lm_head needs every token.
        "lm_head": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),),
                                      use_local_output=False),
    })
    model.lm_head.weight = model.tok_emb.weight  # re-tie to the vocab-sharded DTensor
    distribute_module(model.norm_out, mesh)  # replicated weight

    for block in model.blocks:
        block.attn.n_heads //= tp
        block.attn.n_kv_heads //= tp
        distribute_module(block.norm_attn, mesh)
        distribute_module(block.norm_mlp, mesh)
        parallelize_module(block, mesh, {
            "attn": PrepareModuleInput(input_layouts=(Shard(1), None, None),
                                       desired_input_layouts=(Replicate(), None, None)),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            "mlp": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            "mlp.w_gate": ColwiseParallel(),
            "mlp.w_up": ColwiseParallel(),
            "mlp.w_down": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        })
    return model
