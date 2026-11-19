"""Skip the CUDA tests on GPUs the kernels do not support."""

import pytest
import torch

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu_free: the test does not need a supported GPU"
    )


@pytest.fixture(autouse=True)
def _require_supported_gpu(request):
    if request.node.get_closest_marker("gpu_free") is not None:
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    capability = torch.cuda.get_device_capability()
    if capability not in SUPPORTED_CAPABILITIES:
        pytest.skip("SM100/SM103 required, found SM%d%d" % capability)
