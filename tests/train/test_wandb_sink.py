import pytest

from nanonemo.model.config import ModelConfig
from nanonemo.train.config import TrainConfig
from nanonemo.train.wandb_sink import to_wandb


def test_records_map_to_namespaced_metrics():
    step = {"step": 3, "loss": 9.1, "lr": 6e-4, "grad_norm": 1.2, "tokens": 4096, "tokens_per_s": 6300.0}
    assert to_wandb(step) == {
        "train/loss": 9.1, "train/lr": 6e-4, "train/grad_norm": 1.2,
        "train/tokens": 4096, "perf/tokens_per_s": 6300.0,
    }
    assert to_wandb({"step": 3, "val_loss": 9.0}) == {"val/loss": 9.0}
    assert to_wandb({"event": "start", "params": 1}) == {}


def test_offline_run_writes_history(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    from nanonemo.train.wandb_sink import WandbSink

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = TrainConfig(out_dir=str(tmp_path), wandb_project="nanonemo-test")
    sink = WandbSink(ModelConfig(n_layers=1), cfg)
    sink({"event": "start", "params": 123})
    sink({"step": 0, "loss": 10.9, "lr": 1e-4, "grad_norm": 3.0, "tokens": 1024, "tokens_per_s": 1.0})
    sink({"step": 0, "val_loss": 10.8})
    assert sink.run.config["params"] == 123
    assert sink.run.config["train"]["wandb_project"] == "nanonemo-test"
    sink.finish()
    assert list(tmp_path.glob("wandb/offline-run-*/run-*.wandb"))
