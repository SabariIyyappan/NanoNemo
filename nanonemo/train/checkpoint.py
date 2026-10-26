"""Checkpoints with torch.distributed.checkpoint (DCP).

DCP writes each rank's own shards in parallel, so the same code saves a single-GPU model,
a DDP replica, FSDP2 shards or TP shards, and can load into a different layout (e.g. save
under FSDP on 2 GPUs, resume on 1). get_state_dict / set_state_dict translate between the
wrapped model and plain parameter names.

Layout:  <dir>/step_000123/   DCP files (model + optimizer)
         <dir>/step_000123/trainer.json   step, loader position, loss scale (rank 0 writes it)
         <dir>/latest         name of the newest complete checkpoint
"latest" is written only after a save finishes, so an interrupted save is never resumed from.
"""

import json
import shutil
from concurrent.futures import Future
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def _state(model: nn.Module, opt: torch.optim.Optimizer) -> dict:
    msd, osd = get_state_dict(model, opt)  # also creates empty optimizer state, so a load has a target
    return {"model": msd, "optim": osd}


class Checkpointer:
    def __init__(self, root: str | Path, main: bool, keep: int = 2, use_async: bool = True):
        self.root, self.main, self.keep, self.use_async = Path(root), main, keep, use_async
        self._pending: tuple[Future, Path, dict] | None = None

    def save(self, step: int, model: nn.Module, opt: torch.optim.Optimizer, trainer_state: dict) -> None:
        self.wait()  # one save in flight at a time
        path = self.root / f"step_{step:06d}"
        state = _state(model, opt)
        if self.use_async:
            # Copies the state to CPU, then writes in a background thread while training continues.
            self._pending = (dcp.async_save(state, checkpoint_id=str(path)), path, trainer_state)
        else:
            dcp.save(state, checkpoint_id=str(path))
            self._finish(path, trainer_state)

    def wait(self) -> None:
        if self._pending:
            future, path, trainer_state = self._pending
            future.result()
            self._pending = None
            self._finish(path, trainer_state)

    def _finish(self, path: Path, trainer_state: dict) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # every rank's files are written
        if not self.main:
            return
        (path / "trainer.json").write_text(json.dumps(trainer_state))
        (self.root / "latest").write_text(path.name)
        steps = sorted(p for p in self.root.glob("step_*") if p.is_dir())
        for old in steps[: -self.keep]:
            shutil.rmtree(old, ignore_errors=True)


def find_latest(root: str | Path) -> Path | None:
    """The newest complete checkpoint under root (or root itself if it is one)."""
    root = Path(root)
    if (root / "trainer.json").exists():
        return root
    latest = root / "latest"
    return root / latest.read_text().strip() if latest.exists() else None


def load(path: str | Path, model: nn.Module, opt: torch.optim.Optimizer) -> dict:
    """Load model + optimizer in place; returns the trainer state saved with them."""
    path = Path(path)
    state = _state(model, opt)
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return json.loads((path / "trainer.json").read_text())
