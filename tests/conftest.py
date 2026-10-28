"""pytest -m "not gpu" runs the CPU tests anywhere; the gpu ones (kernels, the TTF update, the script checks) on a
machine with a CUDA GPU."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the tests' training runs record into a throwaway database, not the repository's results/runs.jsonl (before 28 Oct
# they wrote 35 test runs there, 24-25 Oct); set before ttf.results is imported (it reads the paths at import)
_TMP = tempfile.mkdtemp(prefix="ttf_test_results_")
os.environ.setdefault("TTF_RESULTS", os.path.join(_TMP, "runs.jsonl"))
os.environ.setdefault("TTF_NOTES", os.path.join(_TMP, "notes.jsonl"))


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
