"""Reference (S0) training loop: eager FP32, single process, gradient accumulation."""

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
    model = NanoNemo(model_cfg).to(device)
    opt = build_optimizer(model, cfg.peak_lr, cfg.weight_decay)
    train_loader = TokenLoader(cfg.data_dir, "train", cfg.seq_len, cfg.micro_batch, seed=cfg.seed, device=device)
    val_loader = TokenLoader(cfg.data_dir, "val", cfg.seq_len, cfg.micro_batch, seed=cfg.seed, device=device)
    accum, total_steps = cfg.grad_accum(), cfg.total_steps

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

        loss_sum = 0.0
        for _ in range(accum):
            x, y = train_loader.next_batch()
            _, loss = model(x, y)
            # Dividing by accum makes the summed .grad the gradient of the mean loss.
            (loss / accum).backward()
            loss_sum += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)

        is_last = step == total_steps - 1
        if step % cfg.log_every == 0 or is_last:
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            record = {
                "step": step, "loss": loss_sum / accum, "lr": lr, "grad_norm": grad_norm,
                "tokens": (step + 1) * cfg.global_batch_tokens,
                "tokens_per_s": (step - last_logged) * cfg.global_batch_tokens / (now - t_last),
            }
            if device == "cuda":
                record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            t_last, last_logged = now, step
            log(record)
        if (step + 1) % cfg.eval_every == 0 or is_last:
            log({"step": step, "val_loss": evaluate(model, val_loader, cfg.eval_batches)})
            t_last, last_logged = time.perf_counter(), step  # don't count eval time in tokens/s

    metrics_file.close()
    return history
