"""gpu_temp reads this process's GPU (by UUID), not always the first one."""
import types

import torch

from ttf import train


def test_gpu_temp_selects_current_device(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(stdout="57\n")
    monkeypatch.setattr(train.subprocess, "run", run)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(uuid=f"uuid-of-{i}"))
    assert train.gpu_temp() == 57
    assert seen["cmd"][-2:] == ["-i", "GPU-uuid-of-1"]


def test_gpu_temp_none_without_nvidia_smi(monkeypatch):
    def run(cmd, **kw):
        raise FileNotFoundError
    monkeypatch.setattr(train.subprocess, "run", run)
    assert train.gpu_temp() is None
