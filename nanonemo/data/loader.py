"""Deterministic, resumable, distributed-aware token window loader.

The token stream of each shard is cut into windows of seq_len + 1 tokens with
stride seq_len (neighbouring windows share one token): x = window[:-1] and
y = window[1:]. Each epoch visits every window exactly once in an order that
depends only on (seed, epoch).

Sample g (a global index over all ranks) is window perm_epoch[g % n_windows]
with epoch = g // n_windows, where
    g = (micro_step * world_size + rank) * micro_batch + b.
So ranks read disjoint windows, and the whole loader state is `micro_step`.
"""

import json
from pathlib import Path

import numpy as np
import torch

from nanonemo.data.shard import open_shard
from nanonemo.data.tokenizer import TOKENIZER_NAME


class TokenLoader:
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        seq_len: int,
        micro_batch: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
    ):
        data_dir = Path(data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text())
        if manifest["tokenizer"] != TOKENIZER_NAME:
            raise ValueError(f"data tokenized with {manifest['tokenizer']!r}, expected {TOKENIZER_NAME!r}")

        self.shards = [open_shard(data_dir / s["path"]) for s in manifest["splits"][split]["shards"]]
        windows_per_shard = np.array([max(0, (len(s) - 1) // seq_len) for s in self.shards], dtype=np.int64)
        self.n_windows = int(windows_per_shard.sum())
        if self.n_windows == 0:
            raise ValueError(f"split {split!r} has no complete window of {seq_len + 1} tokens; too little data")
        # window w lives in shard searchsorted(ends, w, "right"), at index w - starts[shard]
        self._ends = np.cumsum(windows_per_shard)
        self._starts = self._ends - windows_per_shard

        self.seq_len = seq_len
        self.micro_batch = micro_batch
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.micro_step = 0
        self._perm_epoch = -1
        self._perm: np.ndarray | None = None

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:  # cache one epoch; recomputed from (seed, epoch) alone
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def window_ids(self, micro_step: int) -> np.ndarray:
        """Global window indices for this rank's micro-batch at `micro_step`."""
        first = (micro_step * self.world_size + self.rank) * self.micro_batch
        ids = np.empty(self.micro_batch, dtype=np.int64)
        for b in range(self.micro_batch):
            g = first + b
            ids[b] = self._permutation(g // self.n_windows)[g % self.n_windows]
        return ids

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        windows = np.empty((self.micro_batch, self.seq_len + 1), dtype=np.int64)
        for b, w in enumerate(self.window_ids(self.micro_step)):
            shard = int(np.searchsorted(self._ends, w, side="right"))
            start = int(w - self._starts[shard]) * self.seq_len
            windows[b] = self.shards[shard][start : start + self.seq_len + 1]
        self.micro_step += 1

        batch = torch.from_numpy(windows)
        if self.device.type == "cuda":
            # pinned host memory lets the copy run asynchronously, overlapping GPU compute
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict:
        return {
            "micro_step": self.micro_step,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "micro_batch": self.micro_batch,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("seed", "seq_len", "micro_batch", "world_size"):
            if state[key] != getattr(self, key):
                raise ValueError(f"cannot resume: {key} was {state[key]}, now {getattr(self, key)}")
        self.micro_step = state["micro_step"]
