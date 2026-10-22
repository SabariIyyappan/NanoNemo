import json

import numpy as np
import pytest
import torch

from nanonemo.data.loader import TokenLoader
from nanonemo.data.shard import write_shard

SEQ = 8


def make_dataset(tmp_path, shard_sizes=(100, 57, 230)):
    """Shards hold consecutive integers, so every token tells us exactly where it came from."""
    shards, offset = [], 0
    for i, n in enumerate(shard_sizes):
        info = write_shard(tmp_path / f"train_{i:06d}.bin", np.arange(offset, offset + n, dtype=np.uint16), "gpt2")
        shards.append({"path": f"train_{i:06d}.bin", **info})
        offset += n
    manifest = {"tokenizer": "gpt2", "splits": {"train": {"n_tokens": offset, "shards": shards}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_targets_are_inputs_shifted_by_one(tmp_path):
    loader = TokenLoader(make_dataset(tmp_path), "train", seq_len=SEQ, micro_batch=4)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (4, SEQ) and x.dtype == torch.int64
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    torch.testing.assert_close(y - x, torch.ones_like(x))  # consecutive integers: next token = this + 1


def test_epoch_visits_every_window_once_and_ranks_are_disjoint(tmp_path):
    data = make_dataset(tmp_path)
    ranks = [TokenLoader(data, "train", SEQ, micro_batch=3, seed=1, rank=r, world_size=2) for r in range(2)]
    n_windows = ranks[0].n_windows
    steps = n_windows // (3 * 2)  # full micro-steps within the first epoch
    per_rank = [np.concatenate([loader.window_ids(s) for s in range(steps)]) for loader in ranks]

    assert not set(per_rank[0]) & set(per_rank[1]), "ranks read the same window"
    all_ids = np.concatenate(per_rank)
    assert len(set(all_ids)) == len(all_ids), "a window repeated within one epoch"


def test_windows_never_cross_shard_boundaries(tmp_path):
    loader = TokenLoader(make_dataset(tmp_path), "train", SEQ, micro_batch=5)
    for _ in range(40):
        x, y = loader.next_batch()
        window = torch.cat([x, y[:, -1:]], dim=1)
        # Within one shard the integers are consecutive; a cross-shard window would still be,
        # so check each window lies inside one shard's id range instead.
        lo, hi = window.min(dim=1).values, window.max(dim=1).values
        bounds = [(0, 100), (100, 157), (157, 387)]
        for a, b in zip(lo.tolist(), hi.tolist()):
            assert any(s <= a and b < e for s, e in bounds)


def test_same_seed_same_data_different_seed_different_data(tmp_path):
    data = make_dataset(tmp_path)
    a = TokenLoader(data, "train", SEQ, micro_batch=4, seed=0)
    b = TokenLoader(data, "train", SEQ, micro_batch=4, seed=0)
    c = TokenLoader(data, "train", SEQ, micro_batch=4, seed=1)
    xa, xb, xc = a.next_batch()[0], b.next_batch()[0], c.next_batch()[0]
    torch.testing.assert_close(xa, xb)
    assert not torch.equal(xa, xc)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    data = make_dataset(tmp_path)
    full = TokenLoader(data, "train", SEQ, micro_batch=4, seed=3)
    reference = [full.next_batch()[0] for _ in range(60)]  # 60 steps > 1 epoch, crosses a reshuffle

    first = TokenLoader(data, "train", SEQ, micro_batch=4, seed=3)
    for _ in range(25):
        first.next_batch()
    state = json.loads(json.dumps(first.state_dict()))  # must survive serialization

    resumed = TokenLoader(data, "train", SEQ, micro_batch=4, seed=3)
    resumed.load_state_dict(state)
    for step in range(25, 60):
        torch.testing.assert_close(resumed.next_batch()[0], reference[step])


def test_resume_rejects_changed_config(tmp_path):
    data = make_dataset(tmp_path)
    state = TokenLoader(data, "train", SEQ, micro_batch=4).state_dict()
    with pytest.raises(ValueError, match="micro_batch"):
        TokenLoader(data, "train", SEQ, micro_batch=8).load_state_dict(state)


def test_too_little_data_fails_loudly(tmp_path):
    data = make_dataset(tmp_path, shard_sizes=(5,))
    with pytest.raises(ValueError, match="no complete window"):
        TokenLoader(data, "train", seq_len=SEQ, micro_batch=1)


@pytest.mark.gpu
def test_batch_lands_on_gpu(tmp_path):
    loader = TokenLoader(make_dataset(tmp_path), "train", SEQ, micro_batch=2, device="cuda")
    x, y = loader.next_batch()
    assert x.is_cuda and y.is_cuda
