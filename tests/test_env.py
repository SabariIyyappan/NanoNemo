import json

import pytest

from nanonemo.utils.env import capture_env


def test_capture_env_is_json_serializable():
    env = capture_env()
    json.dumps(env)
    for key in ("torch", "cuda_runtime", "container", "git", "gpu_state"):
        assert key in env


@pytest.mark.gpu
def test_capture_env_sees_gpu():
    env = capture_env()
    assert env["devices"], "no CUDA devices reported"
    assert env["gpu_state"], "nvidia-smi query returned nothing"
    capability = env["devices"][0]["capability"].replace(".", "")
    assert f"sm_{capability}" in env["torch_arch_list"], "torch build does not include this GPU's arch"
