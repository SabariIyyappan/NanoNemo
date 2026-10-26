"""Run config: one YAML file with a `model:` and a `train:` section defines a run."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from nanonemo.model.config import ModelConfig


@dataclass
class TrainConfig:
    data_dir: str = "data/nemotron_cc"
    out_dir: str = "runs/default"
    seed: int = 0
    seq_len: int = 1024
    micro_batch: int = 2
    global_batch_tokens: int = 524_288  # 2^19 ~ 0.5M tokens per optimizer step
    total_tokens: int = 2_500_000_000
    peak_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 50  # optimizer steps
    eval_batches: int = 20
    log_every: int = 1
    precision: str = "fp32"  # fp32 | fp16 (autocast + GradScaler; Turing has no BF16) | bf16 (autocast) | fp8
    fp8_recipe: str = "delayed"  # precision fp8 (BF16 autocast + TE FP8 GEMMs): delayed | current scaling
    compile: str | None = None  # None (eager) | default | max-autotune | reduce-overhead (+ CUDA graphs)
    wandb_project: str | None = None  # None: console + metrics.jsonl only
    wandb_run_name: str | None = None
    # Multi-GPU (launch with torchrun): none | ddp | fsdp | tp. See nanonemo/distributed/.
    parallel: str = "none"
    ddp_bucket_mb: float = 25.0
    ddp_grad_as_bucket_view: bool = False
    fsdp_param_dtype: str = "fp32"  # fp32 | fp16: dtype of the gathered parameters
    fsdp_reshard_after_forward: bool = True
    fsdp_prefetch: bool = False  # explicit next-block all-gather prefetch
    fsdp_compile_blocks: bool = False  # torch.compile each block (instead of `compile` on the whole model)
    # Checkpoints (DCP). Saved every ckpt_every steps (0: never on a schedule), and always on
    # leaving early (stop_after_min / max_steps) or at the end when checkpointing is on.
    ckpt_every: int = 0
    ckpt_dir: str | None = None  # default: <out_dir>/ckpt
    ckpt_async: bool = True
    resume_from: str | None = None  # a checkpoint dir, or a ckpt root with a `latest` file
    stop_after_min: float | None = None  # wall-clock budget for this session (Kaggle limits)
    max_steps: int | None = None  # stop after this many steps in this session

    def __post_init__(self):
        if self.fp8_recipe not in ("delayed", "current"):
            raise ValueError(f"unknown fp8_recipe {self.fp8_recipe!r}")
        if self.precision not in ("fp32", "fp16", "bf16", "fp8"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.compile not in (None, "default", "max-autotune", "reduce-overhead"):
            raise ValueError(f"unknown compile mode {self.compile!r}")
        if self.parallel not in ("none", "ddp", "fsdp", "tp"):
            raise ValueError(f"unknown parallel {self.parallel!r}")
        if self.parallel != "none" and self.compile == "reduce-overhead":
            raise ValueError("CUDA graphs (compile reduce-overhead) are single-GPU only here")
        if self.fsdp_param_dtype not in ("fp32", "fp16"):
            raise ValueError(f"unknown fsdp_param_dtype {self.fsdp_param_dtype!r}")
        if self.fsdp_compile_blocks and (self.parallel != "fsdp" or self.compile is not None):
            raise ValueError("fsdp_compile_blocks needs parallel=fsdp and compile unset")

    @property
    def checkpointing(self) -> bool:
        return bool(self.ckpt_every or self.stop_after_min or self.max_steps or self.resume_from)

    def grad_accum(self, world_size: int = 1) -> int:
        per_micro = self.micro_batch * self.seq_len * world_size
        if self.global_batch_tokens % per_micro:
            raise ValueError(f"global_batch_tokens {self.global_batch_tokens} is not a multiple of {per_micro}")
        return self.global_batch_tokens // per_micro

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.global_batch_tokens

    @property
    def warmup_steps(self) -> int:
        return max(1, round(self.warmup_frac * self.total_steps))


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"model", "train"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return _build(ModelConfig, raw.get("model", {})), _build(TrainConfig, raw.get("train", {}))


def _build(cls, values: dict):
    # A typo in a key would otherwise silently fall back to the default.
    known = {f.name for f in fields(cls)}
    bad = set(values) - known
    if bad:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(bad)}")
    return cls(**values)


def dump_config(model_cfg: ModelConfig, train_cfg: TrainConfig) -> str:
    return yaml.safe_dump({"model": asdict(model_cfg), "train": asdict(train_cfg)}, sort_keys=False)
