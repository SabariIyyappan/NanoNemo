"""Shared microbenchmark helper for the kernel scripts in this folder.

Each script defines, per shape, a dict of providers (eager / compile / triton / liger), each a
function that builds fresh inputs and returns (forward fn, inputs needing .grad reset, outputs->grads).
Timing: each call (forward, or forward + backward) is captured once in a CUDA graph and the
graph is replayed, so the time is GPU time only. Plain triton.testing.do_bench was dominated
by Python autograd/dispatch overhead at these sizes (kernels of 30-300 us), while in training
the GPU never waits on the CPU (see S4), so GPU time is what matters. L2 is flushed before
every replay, and each replay is timed on its own with CUDA events.

Clocks are sampled during every row. On the laptop, platform thermal management can drop the
memory clock from 6801 to 810 MHz (seen in M4) and these kernels slow ~7x with it, so a row
with a slowdown reason or a memory clock >10% under the GPU's max is re-run after a rest.
Achieved bandwidth = the kernel's minimum bytes moved / time, against the GPU's DRAM peak.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import torch

from nanonemo.utils.env import capture_env, gpu_slug
from nanonemo.utils.gpumon import GpuMonitor, summarize
from nanonemo.utils.mfu import peak_gbps


def time_ms(fn, reps: int = 200) -> dict:
    # Warm up on a side stream before capture (PyTorch's CUDA-graph recipe; needed for backward).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    flush = torch.empty(64 * 2**20, dtype=torch.int8, device="cuda")  # 64 MB, well above L2
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for _ in range(10):
        graph.replay()
    for start, end in events:
        flush.zero_()
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in events])
    p20, p50, p80 = torch.quantile(times, torch.tensor([0.2, 0.5, 0.8])).tolist()
    return {"median": p50, "p20": p20, "p80": p80}


def bench_provider(make) -> dict:
    """make() -> (forward_fn, leaf_inputs). forward_fn returns a tuple of outputs."""
    fwd, leaves = make()
    with torch.no_grad():
        fwd_ms = time_ms(fwd)
    grads = [torch.randn_like(o) for o in fwd()]
    # autograd.grad, not .backward(): returns the input gradients without accumulating into .grad.
    return {"fwd_ms": fwd_ms, "fwd_bwd_ms": time_ms(lambda: torch.autograd.grad(fwd(), leaves, grads))}


ATTEMPTS = int(os.environ.get("ATTEMPTS", 3))
REST_S = int(os.environ.get("REST_S", 120))


def max_mem_mhz() -> int:
    out = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=clocks.max.memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def clock_problem(gpu: dict, mem_floor: float) -> str | None:
    if not gpu["n_samples"]:
        return "no GPU samples"
    if gpu["slowdown_reasons"]:
        return ", ".join(gpu["slowdown_reasons"])
    if gpu["mem_mhz_median"] < mem_floor:
        return f"memory clock {gpu['mem_mhz_median']:.0f} MHz"
    return None


def run_row(mon: GpuMonitor, make, mem_floor: float) -> dict:
    for attempt in range(1, ATTEMPTS + 1):
        torch._dynamo.reset()
        t0 = time.monotonic()
        r = bench_provider(make)
        r["gpu"] = summarize(mon.window(t0, time.monotonic()))
        r["flag"] = clock_problem(r["gpu"], mem_floor)
        if r["flag"] is None or attempt == ATTEMPTS:
            return r
        print(f"  flagged ({r['flag']}); resting {REST_S} s before attempt {attempt + 1}")
        time.sleep(REST_S)


def run(name: str, cases: list[dict]) -> None:
    """cases: [{"label", "shape", "bytes_fwd", "bytes_bwd", "providers": {name: make}}]."""
    peak = peak_gbps(torch.cuda.get_device_name())
    mem_floor = 0.9 * max_mem_mhz()
    with GpuMonitor(interval_ms=50) as mon:
        rows = [_row(mon, case, prov, make, peak, mem_floor) for case in cases for prov, make in case["providers"].items()]
    out = Path(f"results/kernels/{gpu_slug()}/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kernel": name, "peak_gbps": peak, "rows": rows, "env": capture_env()}, indent=2) + "\n")
    print(f"saved {out}")


def _row(mon, case, prov, make, peak, mem_floor) -> dict:
    r = run_row(mon, make, mem_floor)
    fwd, fb = r["fwd_ms"]["median"], r["fwd_bwd_ms"]["median"]
    gbps_fwd = case["bytes_fwd"] / fwd / 1e6
    gbps_fb = (case["bytes_fwd"] + case["bytes_bwd"]) / fb / 1e6
    row = {"case": case["label"], "shape": case["shape"], "provider": prov, **r,
           "gbps_fwd": gbps_fwd, "gbps_fwd_bwd": gbps_fb,
           "pct_peak_fwd": gbps_fwd / peak if peak else None,
           "pct_peak_fwd_bwd": gbps_fb / peak if peak else None}
    pct = lambda v: f"{v:5.0%}" if v is not None else "  n/a"  # noqa: E731
    gpu = r["gpu"]
    clocks = f"SM {gpu['sm_mhz']['median']:.0f} / mem {gpu['mem_mhz_median']:.0f} MHz" if gpu["n_samples"] else ""
    print(f"{case['label']:<22} {prov:<8} fwd {fwd * 1e3:7.1f} us {gbps_fwd:6.0f} GB/s {pct(row['pct_peak_fwd'])}"
          f" | fwd+bwd {fb * 1e3:7.1f} us {gbps_fb:6.0f} GB/s {pct(row['pct_peak_fwd_bwd'])} | {clocks}"
          + (f" FLAGGED: {r['flag']}" if r["flag"] else ""))
    return row


def liger_available() -> bool:
    try:
        import liger_kernel  # noqa: F401
    except ImportError:
        return False
    return True
