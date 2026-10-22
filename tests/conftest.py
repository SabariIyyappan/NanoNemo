import pytest
import torch


def pytest_collection_modifyitems(config, items):
    # Tests marked `gpu` / `multigpu` are skipped (not failed) on machines without enough GPUs,
    # so `pytest` runs cleanly on CPU-only CI.
    n_gpus = torch.cuda.device_count()
    for item in items:
        if "gpu" in item.keywords and n_gpus < 1:
            item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU"))
        if "multigpu" in item.keywords and n_gpus < 2:
            item.add_marker(pytest.mark.skip(reason="needs 2+ CUDA GPUs"))
