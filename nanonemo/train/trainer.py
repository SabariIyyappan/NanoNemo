"""Training loop: single process, gradient accumulation, optional AMP and torch.compile."""

import json
import time
from pathlib import Path
from typing import Callable

import torch

from nanonemo.data.loader import TokenLoader
from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo
from nanonemo.train.config import TrainConfig, dump_config
from nanonemo.train.optim import build_optimizer, lr_at
from nanonemo.utils.env import git_info
from nanonemo.utils.mfu import flops_per_token, mfu, peak_tflops


@torch.no_grad()
def evaluate(model: NanoNemo, loader: TokenLoader, n_batches: int) -> float:
    loader.micro_step = 0  # the same val batches every time
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = loader.next_batch()
        total += model(x, y)[1].item()
    model.train()
    return total / n_batches


AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def build_model(model_cfg: ModelConfig, cfg: TrainConfig, device: str):
    """Returns (model, step_model). step_model is the compiled wrapper when cfg.compile is set;
    both share the same parameters. Eval runs on the eager model, so it never triggers a
    recompile (no_grad, different shapes) and needs no CUDA-graph memory of its own."""
    model = NanoNemo(model_cfg).to(device)
    if cfg.compile is None:
        return model, model
    if static_grads(cfg):
        # Under CUDA graphs a backward writes its gradients into the graph's private memory
        # pool, and the next replay reuses that memory. If .grad *were* that tensor (autograd
        # adopts the first micro-batch's gradient as .grad), the second micro-batch would
        # overwrite it. So .grad lives in ordinary memory, allocated once and zeroed in place.
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
    return model, torch.compile(model, mode=None if cfg.compile == "default" else cfg.compile)


def static_grads(cfg: TrainConfig) -> bool:
    return cfg.compile == "reduce-overhead"


def build_scaler(cfg: TrainConfig, device: str) -> torch.amp.GradScaler | None:
    # FP16's max is 65504 and its smallest normal is 6e-5: small gradients underflow to 0.
    # The scaler multiplies the loss up before backward and divides the grads back down,
    # skipping (and shrinking the scale on) any step whose grads overflowed to inf/nan.
    return torch.amp.GradScaler(device) if cfg.precision == "fp16" else None


def train_step(
    model: NanoNemo,
    opt: torch.optim.Optimizer,
    loader: TokenLoader,
    accum: int,
    grad_clip: float,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    keep_grads: bool = False,
):
    """One optimizer step over `accum` micro-batches. Returns (mean loss, pre-clip grad norm).
    keep_grads zeroes .grad in place instead of freeing it (needed with CUDA graphs)."""
    loss_sum = None
    for _ in range(accum):
        x, y = loader.next_batch()
        # CUDA graphs (compile mode reduce-overhead) replay into fixed output buffers.
        # Marking each micro-batch as a new step tells the graph runtime the previous
        # micro-batch's outputs are finished with. A no-op without CUDA graphs.
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, y)
        # Dividing by accum makes the summed .grad the gradient of the mean loss.
        loss = loss / accum
        (scaler.scale(loss) if scaler else loss).backward()
        # Summed on the GPU: a .item() here would stall the CPU on every micro-batch.
        loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
    if scaler:
        scaler.unscale_(opt)  # clip the true gradients, not the scaled ones
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if scaler:
        scaler.step(opt)  # skipped if any grad is inf/nan
        scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=not keep_grads)
    return loss_sum.item(), grad_norm.item()


def train(
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    device: str | None = None,
    log_fn: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Train for cfg.total_steps optimizer steps. Returns the logged records."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(dump_config(model_cfg, cfg))

    torch.manual_seed(cfg.seed)
    model, step_model = build_model(model_cfg, cfg, device)
    opt = build_optimizer(model, cfg.peak_lr, cfg.weight_decay)
    amp_dtype, scaler = AMP_DTYPES[cfg.precision], build_scaler(cfg, device)
    train_loader = TokenLoader(cfg.data_dir, "train", cfg.seq_len, cfg.micro_batch, seed=cfg.seed, device=device)
    val_loader = TokenLoader(cfg.data_dir, "val", cfg.seq_len, cfg.micro_batch, seed=cfg.seed, device=device)
    accum, total_steps = cfg.grad_accum(), cfg.total_steps
    # MFU against the spec's reference peak (FP16 tensor cores), whatever precision this run uses.
    peak = peak_tflops(torch.cuda.get_device_name()) if device == "cuda" else None
    flops_tok = flops_per_token(model_cfg, cfg.seq_len)

    history: list[dict] = []
    metrics_file = (out_dir / "metrics.jsonl").open("w")

    def log(record: dict):
        history.append(record)
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        if log_fn:
            log_fn(record)

    log({
        "event": "start", "git": git_info(), "device": device, "params": model.num_params(),
        "grad_accum": accum, "total_steps": total_steps, "warmup_steps": cfg.warmup_steps,
    })

    t_last, last_logged = time.perf_counter(), -1
    for step in range(total_steps):
        lr = lr_at(step, cfg.peak_lr, total_steps, cfg.warmup_steps, cfg.min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = lr

        loss, grad_norm = train_step(
            step_model, opt, train_loader, accum, cfg.grad_clip, amp_dtype, scaler, static_grads(cfg)
        )

        is_last = step == total_steps - 1
        if step % cfg.log_every == 0 or is_last:
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            record = {
                "step": step, "loss": loss, "lr": lr, "grad_norm": grad_norm,
                "tokens": (step + 1) * cfg.global_batch_tokens,
                "tokens_per_s": (step - last_logged) * cfg.global_batch_tokens / (now - t_last),
            }
            if device == "cuda":
                record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if scaler:
                record["loss_scale"] = scaler.get_scale()
            if peak:
                record["mfu"] = mfu(record["tokens_per_s"], flops_tok, peak["fp16_tc"])
            t_last, last_logged = now, step
            log(record)
        if (step + 1) % cfg.eval_every == 0 or is_last:
            with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
                val_loss = evaluate(model, val_loader, cfg.eval_batches)
            log({"step": step, "val_loss": val_loss})
            t_last, last_logged = time.perf_counter(), step  # don't count eval time in tokens/s

    metrics_file.close()
    return history
