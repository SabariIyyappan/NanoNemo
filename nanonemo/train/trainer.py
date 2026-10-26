"""Training loop: gradient accumulation, AMP, torch.compile, multi-GPU (DDP / FSDP2 / TP), checkpoints."""

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager

import torch
from torch import nn

from nanonemo import distributed
from nanonemo.data.loader import TokenLoader
from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo
from nanonemo.train import checkpoint
from nanonemo.train.config import TrainConfig, dump_config
from nanonemo.train.optim import build_optimizer, lr_at
from nanonemo.utils.env import git_info
from nanonemo.utils.mfu import flops_per_token, mfu, peak_tflops


@torch.no_grad()
def evaluate(model: NanoNemo, loader: TokenLoader, n_batches: int, loss_ctx: Callable[[], ContextManager] = nullcontext) -> float:
    loader.micro_step = 0  # the same val batches every time
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = loader.next_batch()
        with loss_ctx():
            total += model(x, y)[1].item()
    model.train()
    return total / n_batches


# fp8: BF16 autocast for everything, plus FP8 GEMMs in the Transformer Engine linears.
AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}


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


@dataclass
class Setup:
    """How a run's model is laid out over the GPUs."""

    model: nn.Module  # owns the parameters: optimizer, checkpoints, eval
    step_model: nn.Module  # what the training step calls (compiled and/or wrapped)
    dp_rank: int = 0  # which slice of the data this rank reads
    dp_world: int = 1
    # sync_ctx(last): wraps a micro-batch's forward + backward; gradients are synced across
    # ranks only on the last micro-batch of a step (DDP no_sync / FSDP requires_gradient_sync).
    sync_ctx: Callable[[bool], ContextManager] = lambda last: nullcontext()
    loss_ctx: Callable[[], ContextManager] = nullcontext  # TP: loss_parallel()
    fwd_ctx: Callable[[], ContextManager] = nullcontext  # around the forward only: FP8 autocast
    sharded_grads: bool = False  # gradients are DTensors (FSDP, TP)


def build_parallel(model_cfg: ModelConfig, cfg: TrainConfig, d: distributed.Dist) -> Setup:
    setup = _build_parallel(model_cfg, cfg, d)
    if cfg.precision == "fp8":
        if model_cfg.linear_impl != "te":
            raise ValueError("precision fp8 needs model.linear_impl: te")
        from nanonemo.te import fp8_context

        setup.fwd_ctx = fp8_context(cfg.fp8_recipe)
    return setup


def _build_parallel(model_cfg: ModelConfig, cfg: TrainConfig, d: distributed.Dist) -> Setup:
    if cfg.parallel == "none":
        model, step_model = build_model(model_cfg, cfg, d.device)
        return Setup(model, step_model)

    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh(d.device.split(":")[0], (d.world,))
    model = NanoNemo(model_cfg).to(d.device)
    compile_mode = None if cfg.compile == "default" else cfg.compile

    def compiled(m):
        return torch.compile(m, mode=compile_mode) if cfg.compile else m

    if cfg.parallel == "ddp":
        from nanonemo.distributed import ddp

        wrapped = ddp.wrap(model, d.device, cfg.ddp_bucket_mb, cfg.ddp_grad_as_bucket_view)
        return Setup(model, compiled(wrapped), d.rank, d.world,
                     sync_ctx=lambda last: nullcontext() if last else wrapped.no_sync())
    if cfg.parallel == "fsdp":
        from nanonemo.distributed import fsdp

        fsdp.shard(model, mesh, cfg.fsdp_param_dtype, cfg.fsdp_reshard_after_forward, cfg.fsdp_prefetch,
                   cfg.fsdp_compile_blocks)

        def sync_ctx(last):
            model.set_requires_gradient_sync(last)  # reduce-scatter only once per step
            return nullcontext()

        return Setup(model, compiled(model), d.rank, d.world, sync_ctx=sync_ctx, sharded_grads=True)
    from torch.distributed.tensor.parallel import loss_parallel

    from nanonemo.distributed import tp

    tp.parallelize(model, mesh)
    if cfg.compile:
        # Per block: the loss runs on vocab-sharded DTensors under loss_parallel(), which a
        # whole-model graph gets wrong in backward (the logits' gradient comes back as a
        # plain tensor). Blocks hold all the matmuls, so this keeps nearly all the benefit.
        for block in model.blocks:
            block.compile(mode=compile_mode)
    # Every TP rank works on the same batch: it is one data-parallel replica.
    return Setup(model, model, loss_ctx=loss_parallel, sharded_grads=True)


def build_scaler(cfg: TrainConfig, device: str, sharded: bool = False) -> torch.amp.GradScaler | None:
    # FP16's max is 65504 and its smallest normal is 6e-5: small gradients underflow to 0.
    # The scaler multiplies the loss up before backward and divides the grads back down,
    # skipping (and shrinking the scale on) any step whose grads overflowed to inf/nan.
    if cfg.precision != "fp16":
        return None
    if sharded:
        # Each rank sees only its shard of the gradients, so "any inf?" must be agreed on
        # across ranks, or one rank would skip a step the others take.
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        return ShardedGradScaler(device.split(":")[0])
    return torch.amp.GradScaler(device.split(":")[0])


def train_step(
    model: NanoNemo,
    opt: torch.optim.Optimizer,
    loader: TokenLoader,
    accum: int,
    grad_clip: float,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    keep_grads: bool = False,
    sync_ctx: Callable[[bool], ContextManager] = lambda last: nullcontext(),
    loss_ctx: Callable[[], ContextManager] = nullcontext,
    fwd_ctx: Callable[[], ContextManager] = nullcontext,
):
    """One optimizer step over `accum` micro-batches. Returns (mean loss over ranks, pre-clip grad norm).
    keep_grads zeroes .grad in place instead of freeing it (needed with CUDA graphs)."""
    loss_sum = None
    for i in range(accum):
        x, y = loader.next_batch()
        # CUDA graphs (compile mode reduce-overhead) replay into fixed output buffers.
        # Marking each micro-batch as a new step tells the graph runtime the previous
        # micro-batch's outputs are finished with. A no-op without CUDA graphs.
        torch.compiler.cudagraph_mark_step_begin()
        with sync_ctx(i == accum - 1), loss_ctx():
            with torch.autocast(x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None), fwd_ctx():
                _, loss = model(x, y)
            # Dividing by accum makes the summed .grad the gradient of the mean loss.
            loss = loss / accum
            (scaler.scale(loss) if scaler else loss).backward()
        # Summed on the GPU: a .item() here would stall the CPU on every micro-batch.
        loss = loss.detach()
        if hasattr(loss, "full_tensor"):
            loss = loss.full_tensor()
        loss_sum = loss if loss_sum is None else loss_sum + loss
    if scaler:
        scaler.unscale_(opt)  # clip the true gradients, not the scaled ones
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()  # a DTensor under FSDP / TP
    if scaler:
        scaler.step(opt)  # skipped if any grad is inf/nan
        scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=not keep_grads)
    return distributed.all_mean(loss_sum).item(), grad_norm.item()


def train(
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    device: str | None = None,
    log_fn: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Train until cfg.total_steps optimizer steps (or a session limit). Returns rank 0's logged records.

    Under torchrun every rank calls this; only rank 0 writes files and logs.
    """
    d = distributed.init()
    if cfg.parallel != "none" and not d.enabled:
        raise ValueError(f"parallel={cfg.parallel!r} needs several processes: launch with torchrun")
    if cfg.parallel == "none" and d.world > 1:
        raise ValueError(f"{d.world} processes but parallel='none': set ddp, fsdp or tp")
    if device is not None and not d.enabled:
        d = distributed.Dist(device=device)
    device = d.device
    out_dir = Path(cfg.out_dir)
    if d.main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(dump_config(model_cfg, cfg))

    torch.manual_seed(cfg.seed)
    setup = build_parallel(model_cfg, cfg, d)
    model = setup.model
    opt = build_optimizer(model, cfg.peak_lr, cfg.weight_decay)
    amp_dtype, scaler = AMP_DTYPES[cfg.precision], build_scaler(cfg, device, setup.sharded_grads)
    train_loader = TokenLoader(cfg.data_dir, "train", cfg.seq_len, cfg.micro_batch, seed=cfg.seed,
                               rank=setup.dp_rank, world_size=setup.dp_world, device=device)
    val_loader = TokenLoader(cfg.data_dir, "val", cfg.seq_len, cfg.micro_batch, seed=cfg.seed, device=device)
    accum, total_steps = cfg.grad_accum(setup.dp_world), cfg.total_steps
    # MFU against the spec's reference peak (FP16 tensor cores), whatever precision this run uses.
    on_cuda = device.startswith("cuda")
    peak = peak_tflops(torch.cuda.get_device_name()) if on_cuda else None
    flops_tok = flops_per_token(model_cfg, cfg.seq_len)

    ckpt = None
    if cfg.checkpointing:
        ckpt = checkpoint.Checkpointer(cfg.ckpt_dir or out_dir / "ckpt", d.main, use_async=cfg.ckpt_async)
    start_step = 0
    resumed_from = None
    if cfg.resume_from:
        path = checkpoint.find_latest(cfg.resume_from)
        if path is None:
            raise FileNotFoundError(f"no checkpoint under {cfg.resume_from}")
        state = checkpoint.load(path, model, opt)
        # micro_step counts per-rank micro-batches; convert through the global count so a
        # run can resume on a different number of data-parallel ranks.
        consumed = state["micro_step"] * state.get("dp_world", 1)
        if consumed % setup.dp_world:
            raise ValueError(f"cannot split {consumed} consumed micro-batches over {setup.dp_world} ranks")
        start_step, train_loader.micro_step = state["step"] + 1, consumed // setup.dp_world
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        resumed_from = str(path)

    def trainer_state(step: int) -> dict:
        return {"step": step, "micro_step": train_loader.micro_step,
                "scaler": scaler.state_dict() if scaler else None, "world": d.world, "dp_world": setup.dp_world,
                "parallel": cfg.parallel}

    history: list[dict] = []
    metrics_file = (out_dir / "metrics.jsonl").open("a" if resumed_from else "w") if d.main else None

    def log(record: dict):
        if not d.main:
            return
        history.append(record)
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        if log_fn:
            log_fn(record)

    log({
        "event": "start", "git": git_info(), "device": device, "params": model.num_params(),
        "grad_accum": accum, "total_steps": total_steps, "warmup_steps": cfg.warmup_steps,
        "parallel": cfg.parallel, "world": d.world, "start_step": start_step, "resumed_from": resumed_from,
    })

    t_session = t_last = time.perf_counter()
    last_logged = start_step - 1
    for step in range(start_step, total_steps):
        lr = lr_at(step, cfg.peak_lr, total_steps, cfg.warmup_steps, cfg.min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = lr

        loss, grad_norm = train_step(
            setup.step_model, opt, train_loader, accum, cfg.grad_clip, amp_dtype, scaler, static_grads(cfg),
            setup.sync_ctx, setup.loss_ctx, setup.fwd_ctx,
        )

        is_last = step == total_steps - 1
        if step % cfg.log_every == 0 or is_last:
            if on_cuda:
                torch.cuda.synchronize()
            now = time.perf_counter()
            record = {
                "step": step, "loss": loss, "lr": lr, "grad_norm": grad_norm,
                "tokens": (step + 1) * cfg.global_batch_tokens,
                "tokens_per_s": (step - last_logged) * cfg.global_batch_tokens / (now - t_last),
            }
            if on_cuda:
                record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if scaler:
                record["loss_scale"] = scaler.get_scale()
            if peak:
                # Per-GPU MFU: the global token rate shared over all GPUs.
                record["mfu"] = mfu(record["tokens_per_s"] / d.world, flops_tok, peak["fp16_tc"])
            t_last, last_logged = now, step
            log(record)
        if (step + 1) % cfg.eval_every == 0 or is_last:
            with torch.autocast(device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
                val_loss = evaluate(model, val_loader, cfg.eval_batches, setup.loss_ctx)
            log({"step": step, "val_loss": val_loss})
            t_last, last_logged = time.perf_counter(), step  # don't count eval time in tokens/s

        if ckpt:
            out_of_time = cfg.stop_after_min is not None and time.perf_counter() - t_session > 60 * cfg.stop_after_min
            # Every rank must agree, or one would stop and leave the others waiting in a collective.
            stop = distributed.any_rank(out_of_time, device) or (
                cfg.max_steps is not None and step - start_step + 1 >= cfg.max_steps)
            if stop or is_last or (cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0):
                ckpt.save(step, model, opt, trainer_state(step))
                log({"step": step, "event": "checkpoint"})
                t_last, last_logged = time.perf_counter(), step
            if stop and not is_last:
                log({"step": step, "event": "stopped", "reason": "time limit" if out_of_time else "max_steps"})
                break

    if ckpt:
        ckpt.wait()
    if metrics_file:
        metrics_file.close()
    return history
