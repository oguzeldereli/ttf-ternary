"""Validation loss as the trainer measures it (random windows, fixed seed)."""
from __future__ import annotations
import numpy as np
import torch

from .bitlinear import STATE
from .data import get_batch


@torch.no_grad()
def evaluate(model, data, tc, device):
    model.eval()
    STATE.updates_enabled = False
    losses = []
    # fixed windows: same val batches every eval and across runs (comparable). At least 16 windows per batch whatever
    # the training batch size (before 5 Oct a run at batch 4 was scored on the first quarter of a batch-16 run's windows,
    # so training-time val losses of different batch sizes were not comparable), run in chunks of the batch size
    gen = torch.Generator().manual_seed(tc.seed + 12345)
    bs = max(tc.batch_size, 16)
    for _ in range(tc.eval_iters):
        x, y = get_batch(data, bs, tc.seq_len, device, gen)
        for a in range(0, bs, tc.batch_size):
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x[a:a + tc.batch_size], y[a:a + tc.batch_size])
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))
