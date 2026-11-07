"""ple_pq.py: the Q38 PLE n-gram table read from its compressed store (prune + product quantization), decoded on the GPU
in the training loop instead of the 51.2 GB FP8 table. The store is built by ttfconv (ple_pq build, with the fits of
ple_loss).

Stored form (DIR):
  codes.bin   uint8 [total, 160 / D]: per table row, the codebook index of each D-dim sub-vector of P_h x (P_h: head h's
              value_proj metric basis); rows not kept are 0
  keep.bin    uint8 [total]: 1 where the row was hit >= N times in the count windows (prune N), else the row reads as 0
  meta.pt     {"D", "K", "N", "count_windows", "offs", "total", "Pi" [heads, 160, 160], "cb" [heads, 160 / D, K, D]}
A row decodes as keep * Pi_h (cb_h[j, code_j])_j (float32 on the GPU).
Use: python -m ttf.q38.train ... --ple-pq DIR"""
import os
import numpy as np
import torch


class PQTable:
    """NGramEmbedding.table stand-in: table[rows] -> float32 [n, dim] on `device` (scale applied: set ng.scale = None)"""

    def __init__(self, d, device="cuda"):
        m = torch.load(os.path.join(d, "meta.pt"), weights_only=False)
        self.D, self.K, self.total = m["D"], m["K"], m["total"]
        self.nS = m["cb"].shape[1]
        self.dev = torch.device(device)
        self.offs = torch.tensor(list(m["offs"]) + [self.total], dtype=torch.long)
        self.Pi = m["Pi"].float().to(self.dev)                   # [heads, 160, 160]
        self.cb = m["cb"].float().to(self.dev)                   # [heads, nS, K, D]
        self.codes = torch.from_numpy(np.fromfile(os.path.join(d, "codes.bin"), dtype=np.uint8).reshape(self.total, self.nS))
        self.keep = torch.from_numpy(np.fromfile(os.path.join(d, "keep.bin"), dtype=np.uint8))
        assert self.keep.numel() == self.total
        self.ar = torch.arange(self.nS, device=self.dev)

    def __getitem__(self, rows):
        r = rows.reshape(-1).cpu().long()
        c = self.codes.index_select(0, r).to(self.dev, non_blocking=True).long()
        k = self.keep.index_select(0, r).to(self.dev, non_blocking=True).float()
        h = (torch.searchsorted(self.offs, r, right=True) - 1).to(self.dev)
        out = torch.empty(len(r), self.Pi.shape[1], device=self.dev)
        for hh in torch.unique(h).tolist():
            sel = (h == hh).nonzero().squeeze(1)
            X = self.cb[hh][self.ar[None], c[sel]].reshape(len(sel), -1)     # [m, nS, D] -> [m, 160]
            out[sel] = X @ self.Pi[hh].T
        return out * k[:, None]
