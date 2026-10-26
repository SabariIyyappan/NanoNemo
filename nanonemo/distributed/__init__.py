"""Process-group setup for torchrun, and the three parallel layouts (ddp.py, fsdp.py, tp.py).

Launched by torchrun, every process reads RANK / WORLD_SIZE / LOCAL_RANK from the environment
and joins one process group. Without torchrun this is a single process and nothing is set up.
NANONEMO_DEVICE=cpu forces CPU + gloo, which lets the multi-process tests run on one machine
without two GPUs (NCCL refuses two ranks on one GPU).
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    device: str = "cuda"

    @property
    def main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return dist.is_available() and dist.is_initialized()


def init() -> Dist:
    device_type = os.environ.get("NANONEMO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if "RANK" not in os.environ:
        return Dist(device=device_type)
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if dist.is_initialized():  # already set up by an earlier call in this process
        return Dist(rank, world, local, f"cuda:{local}" if device_type == "cuda" else "cpu")
    if device_type == "cuda":
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
        # gloo next to NCCL: async checkpoint saves stage on the CPU and need a CPU backend.
        dist.init_process_group("cuda:nccl,cpu:gloo", device_id=torch.device(device))
    else:
        device = "cpu"
        dist.init_process_group("gloo")
    return Dist(rank, world, local, device)


def destroy() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean of a tensor over all ranks (identity in a single process)."""
    if dist.is_available() and dist.is_initialized():
        x = x.clone()
        dist.all_reduce(x)
        x /= dist.get_world_size()
    return x


def any_rank(flag: bool, device: str) -> bool:
    """True if `flag` is true on any rank, so all ranks take the same branch."""
    if not (dist.is_available() and dist.is_initialized()):
        return flag
    t = torch.tensor(float(flag), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())
