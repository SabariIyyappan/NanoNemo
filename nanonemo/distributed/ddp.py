"""Data parallel: every rank holds the full model; gradients are all-reduced in buckets.

DDP launches an all-reduce for a bucket as soon as all of its gradients are ready, so
communication overlaps the rest of the backward. bucket_cap_mb trades overlap (small
buckets start earlier) against per-call overhead (fewer, larger calls). With
gradient_as_bucket_view, .grad is a view into the bucket, which saves one copy and one
model-sized buffer.
"""

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def wrap(model: nn.Module, device: str, bucket_cap_mb: float, gradient_as_bucket_view: bool) -> DDP:
    device_ids = [int(device.split(":")[1])] if device.startswith("cuda:") else None
    return DDP(model, device_ids=device_ids, bucket_cap_mb=bucket_cap_mb, gradient_as_bucket_view=gradient_as_bucket_view)
