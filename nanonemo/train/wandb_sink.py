"""Forward trainer records to Weights & Biases."""

from dataclasses import asdict

from nanonemo.model.config import ModelConfig
from nanonemo.train.config import TrainConfig

_KEYS = {
    "loss": "train/loss",
    "lr": "train/lr",
    "grad_norm": "train/grad_norm",
    "tokens": "train/tokens",
    "tokens_per_s": "perf/tokens_per_s",
    "peak_mem_gb": "perf/peak_mem_gb",
    "mfu": "perf/mfu",
    "val_loss": "val/loss",
}


def to_wandb(record: dict) -> dict:
    """Map a trainer record to namespaced W&B metric names (unknown fields are dropped)."""
    return {_KEYS[k]: v for k, v in record.items() if k in _KEYS}


class WandbSink:
    def __init__(self, model_cfg: ModelConfig, cfg: TrainConfig):
        import wandb  # only needed when W&B logging is on

        self._wandb = wandb
        self.run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            dir=cfg.out_dir,
            config={"model": asdict(model_cfg), "train": asdict(cfg)},
        )

    def __call__(self, record: dict) -> None:
        if record.get("event") == "start":
            self.run.config.update({k: v for k, v in record.items() if k != "event"}, allow_val_change=True)
            return
        metrics = to_wandb(record)
        if metrics:
            self.run.log(metrics, step=record["step"])

    def finish(self) -> None:
        self.run.finish()
