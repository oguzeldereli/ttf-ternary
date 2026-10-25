"""Low-memory AdamW for the float tail (the token embedding + norms).

States m, v are kept in bfloat16 (2 bytes each) instead of fp32 (4 each), so the
tail optimizer costs 4 bytes/param instead of 8 — half — while staying a correct,
standard Adam. (An earlier int8-blockwise version was numerically unstable and
was removed.) Class name kept as Adam8bit for import compatibility.
"""
from __future__ import annotations
import torch


CHUNK = 1 << 24


class Adam8bit(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for grp in self.param_groups:
            b1, b2 = grp["betas"]
            lr, eps, wd = grp["lr"], grp["eps"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    st["v"] = torch.zeros_like(p, dtype=torch.bfloat16)
                st["step"] += 1
                bc1 = 1 - b1 ** st["step"]
                bc2 = 1 - b2 ** st["step"]
                if wd > 0:
                    p.mul_(1 - lr * wd)
                # in chunks of CHUNK elements: the fp32 temporaries (g, m, v, the denominator) of a whole large
                # parameter (a 27B embedding: 1.3G elements, ~5 GiB each) were the step's memory peak. Elementwise,
                # so the chunks give the same values.
                ts = (p, p.grad, st["m"], st["v"])
                flat = all(t.is_contiguous() for t in ts)       # (else one piece: a view would fail, a reshape copy)
                pf, gf, mf, vf = (t.view(-1) for t in ts) if flat else ts
                for c in ([slice(i, i + CHUNK) for i in range(0, pf.numel(), CHUNK)] if flat else [slice(None)]):
                    g = gf[c].float()
                    m = mf[c].float().mul_(b1).add_(g, alpha=1 - b1)
                    v = vf[c].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                    mf[c].copy_(m)                      # store back as bf16
                    vf[c].copy_(v)
                    denom = (v / bc2).sqrt_().add_(eps)
                    pf[c].addcdiv_(m / bc1, denom, value=-lr)
        return loss
