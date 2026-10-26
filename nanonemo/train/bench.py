"""Training-throughput benchmark: warmup excluded, N timed steps x K trials, median + min/max.

Times the real optimizer step (trainer.train_step) on real data. Each step is bracketed
by CUDA events, so the timing covers the GPU work itself, not just the kernel launches.
GPU clocks and temperature are sampled in the background, and trials whose clocks
dropped are flagged, since a throttled laptop run is not comparable with a cool one.
"""

import os
import time
from dataclasses import asdict
from statistics import median

import torch

from nanonemo import distributed
from nanonemo.data.loader import TokenLoader
from nanonemo.model.config import ModelConfig
from nanonemo.train.config import TrainConfig
from nanonemo.train.optim import build_optimizer, lr_at
from nanonemo.train.trainer import AMP_DTYPES, build_parallel, build_scaler, static_grads, train_step
from nanonemo.utils.env import capture_env
from nanonemo.utils.gpumon import GpuMonitor, summarize, throttle_flags
from nanonemo.utils.mfu import flops_per_token, mfu, peak_tflops, rated_boost_mhz


def _spread(values: list[float]) -> dict:
    return {"median": median(values), "min": min(values), "max": max(values)}


class _StepTimer:
    """Per-step durations in ms: CUDA events on the GPU, perf_counter on the CPU."""

    def __init__(self, n_steps: int, cuda: bool):
        self.cuda = cuda
        if cuda:
            self.marks = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps + 1)]
        else:
            self.marks = [0.0] * (n_steps + 1)

    def mark(self, i: int):
        if self.cuda:
            self.marks[i].record()
        else:
            self.marks[i] = time.perf_counter()

    def step_ms(self) -> list[float]:
        if self.cuda:
            torch.cuda.synchronize()  # elapsed_time needs both events to have completed
            return [a.elapsed_time(b) for a, b in zip(self.marks, self.marks[1:])]
        return [(b - a) * 1e3 for a, b in zip(self.marks, self.marks[1:])]


def benchmark(
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    steps: int = 50,
    trials: int = 3,
    warmup_steps: int = 5,
    device: str | None = None,
    monitor_interval_ms: int = 200,
) -> dict:
    """Under torchrun (cfg.parallel != 'none') every rank runs this; the result is rank 0's view."""
    d = distributed.init() if device is None else distributed.Dist(device=device)
    device = d.device
    cuda = device.startswith("cuda")
    env = capture_env()  # before the run: idle clocks/temperature at the start

    torch.manual_seed(cfg.seed)
    setup = build_parallel(model_cfg, cfg, d)
    model, step_model = setup.model, setup.step_model
    opt = build_optimizer(model, cfg.peak_lr, cfg.weight_decay)
    amp_dtype, scaler = AMP_DTYPES[cfg.precision], build_scaler(cfg, device, setup.sharded_grads)
    loader = TokenLoader(cfg.data_dir, "train", cfg.seq_len, cfg.micro_batch, seed=cfg.seed,
                         rank=setup.dp_rank, world_size=setup.dp_world, device=device)
    accum = cfg.grad_accum(setup.dp_world)
    # A short schedule of its own: the LR doesn't change the speed, but a warmup keeps
    # the loss sane (no FP16 overflow/skip storms in later stages).
    total = warmup_steps + steps * trials
    lr_warmup = max(1, round(cfg.warmup_frac * total))
    step = 0

    def run_step():
        nonlocal step
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg.peak_lr, total, lr_warmup, cfg.min_lr_ratio)
        step += 1
        return train_step(step_model, opt, loader, accum, cfg.grad_clip, amp_dtype, scaler, static_grads(cfg),
                          setup.sync_ctx, setup.loss_ctx)

    # Warmup: first CUDA context use, cuBLAS heuristics, allocator growth, and (later stages)
    # torch.compile and CUDA-graph capture. Timed, but kept out of the throughput numbers.
    t0 = time.perf_counter()
    for _ in range(warmup_steps):
        run_step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    warmup_s = time.perf_counter() - t0

    # NANONEMO_PROFILE=1: open the profiler capture window (nsys --capture-range=cudaProfilerApi)
    # for the first trial only, so the trace holds steady-state steps and nothing else.
    profile = os.environ.get("NANONEMO_PROFILE") == "1" and cuda
    trial_results = []
    with GpuMonitor(monitor_interval_ms) as mon:  # records nothing without nvidia-smi
        for trial in range(trials):
            if profile and trial == 0:
                torch.cuda.profiler.start()
            timer = _StepTimer(steps, cuda)
            losses = []
            t_start = time.monotonic()
            timer.mark(0)
            for i in range(steps):
                losses.append(run_step()[0])
                timer.mark(i + 1)
            step_ms = timer.step_ms()
            if profile and trial == 0:
                torch.cuda.profiler.stop()
            t_end = time.monotonic()
            trial_s = sum(step_ms) / 1e3
            trial_results.append({
                "tokens_per_s": steps * cfg.global_batch_tokens / trial_s,
                "step_ms": step_ms,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_scale": scaler.get_scale() if scaler else None,
                "gpu": summarize(mon.window(t_start, t_end)),
            })

    tps = [t["tokens_per_s"] for t in trial_results]
    flops_tok = flops_per_token(model_cfg, cfg.seq_len)
    peak = peak_tflops(torch.cuda.get_device_name()) if cuda else None
    return {
        "bench": {"steps": steps, "trials": trials, "warmup_steps": warmup_steps, "grad_accum": accum,
                  "tokens_per_step": cfg.global_batch_tokens, "device": device, "world": d.world,
                  "parallel": cfg.parallel},
        "tokens_per_s": _spread(tps),
        "step_ms": _spread([median(t["step_ms"]) for t in trial_results]),
        "flops_per_token": flops_tok,
        "achieved_tflops": median(tps) * flops_tok / 1e12,
        "peak_tflops": peak,
        # MFU per reference peak and per GPU (the global token rate shared over all GPUs);
        # "fp16_tc" is the spec's definition.
        "mfu": {k: _spread([mfu(x / d.world, flops_tok, v) for x in tps]) for k, v in peak.items()} if peak else None,
        "peak_mem_gb": {
            "allocated": torch.cuda.max_memory_allocated() / 2**30,
            "reserved": torch.cuda.max_memory_reserved() / 2**30,
        } if cuda else None,
        "warmup_s": warmup_s,
        "throttle_flags": throttle_flags(
            [t["gpu"] for t in trial_results], min_mhz=rated_boost_mhz(torch.cuda.get_device_name())
        ) if cuda else [],
        "trials": trial_results,
        "params": model.num_params(),
        "config": {"model": asdict(model_cfg), "train": asdict(cfg)},
        "env": env,
    }

