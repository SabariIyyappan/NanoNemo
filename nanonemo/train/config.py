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
    wandb_project: str | None = None  # None: console + metrics.jsonl only
    wandb_run_name: str | None = None

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
