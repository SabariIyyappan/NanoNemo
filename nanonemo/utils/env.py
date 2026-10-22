"""Capture the software/hardware environment that every result is recorded with."""

import datetime
import os
import platform
import socket
import subprocess


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


# Live GPU state from nvidia-smi. Clocks and temperature matter on a laptop:
# a throttled run is not comparable with an unthrottled one.
_SMI_FIELDS = [
    "name",
    "driver_version",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks_throttle_reasons.active",
]


def gpu_state() -> list[dict]:
    out = _run(["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"])
    if out is None:
        return []
    return [dict(zip(_SMI_FIELDS, (v.strip() for v in line.split(",")))) for line in out.splitlines()]


def capture_env() -> dict:
    import torch

    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None

    cuda = torch.cuda.is_available()
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if cuda else None,
        "container": os.environ.get("NVIDIA_PYTORCH_VERSION"),
        "torch_arch_list": torch.cuda.get_arch_list() if cuda else [],
        "devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
            }
            for i in range(torch.cuda.device_count())
        ],
        "gpu_state": gpu_state(),
        "git": git_info(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(capture_env(), indent=2))
