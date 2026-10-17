"""pytest -m "not gpu" runs the CPU tests anywhere; the gpu ones (kernels, the TTF update, the script checks) on a lab
PC with a CUDA GPU (never the laptop's)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU")


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        skip = pytest.mark.skip(reason="no CUDA GPU")
        for it in items:
            if "gpu" in it.keywords:
                it.add_marker(skip)
