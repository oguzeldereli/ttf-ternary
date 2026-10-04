"""--lean_tail: AdamW for the float tail (embedding, adapter, norm gains, row / column scales) with low-precision storage.

- Parameters stay in their own low-precision dtype (bf16 for the embedding and the adapter, fp16 for gains and scales);
  the update is computed in fp32 and written back with stochastic rounding (uniform noise of one unit in the last place
  before rounding), so steps smaller than half a unit are kept on average instead of rounding away (the reason the
  bf16 tail once froze the norm gains at 1.0).
- First moment: int8 codes with one fp32 scale per row (2-D) or per tensor (1-D), stochastic rounding; the step uses the
  unrounded moment, the rounded one is stored (as the int8 short momentum of --ts).
- Second moment: factored for 2-D parameters (row and column means of g^2, as --ts's v), fp32 for 1-D ones (tiny).
Bytes per 2-D parameter: 2 (value) + 2 (gradient) + 1 (first moment) + ~0 (second moment), against 16 for fp32 AdamW.
"""
from __future__ import annotations
import torch

_MANT = {torch.bfloat16: 7, torch.float16: 10}


def stochastic_round(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """fp32 -> dtype, rounding up or down with probability given by the distance (exact zeros stay zero)"""
    if dtype == torch.float32:
        return x
    _, e = torch.frexp(x)
    ulp = torch.ldexp(torch.ones_like(x), e - 1 - _MANT[dtype])
    ulp = torch.where(x == 0, torch.zeros_like(ulp), ulp)
    return (x + (torch.rand_like(x) - 0.5) * ulp).to(dtype)


class LeanAdam(torch.optim.Optimizer):
    CHUNK = 2048          # rows per block of a 2-D update

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def load_state_dict(self, state_dict):
        """torch's Optimizer.load_state_dict casts the state to the parameter's dtype (bf16 / fp16 here): the int8 first
        moment becomes bf16 (2x the memory), the row scales and factored second moment are rounded to bf16, and the fp16
        gains' second moment (~1e-10) is flushed to zero (NaN on the first step after a resume). Put the saved state back
        in its own dtypes."""
        super().load_state_dict(state_dict)
        params = [p for g in self.param_groups for p in g["params"]]
        for i, st in state_dict["state"].items():
            p = params[int(i)]
            for k, v in st.items():
                if torch.is_tensor(v) and k != "step":
                    self.state[p][k] = v.to(device=p.device, copy=True)

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
                two_d = p.dim() == 2
                if not st:
                    st["step"] = 0
                    if two_d:
                        st["mq"] = torch.zeros_like(p, dtype=torch.int8)
                        st["ms"] = torch.zeros(p.shape[0], 1, device=p.device)
                        st["R"] = torch.zeros(p.shape[0], device=p.device)
                        st["C"] = torch.zeros(p.shape[1], device=p.device)
                    else:
                        st["m"] = torch.zeros_like(p, dtype=torch.float32)
                        st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                if two_d:
                    # second moment first (it needs the mean over all rows), then the rest in blocks of rows so the
                    # fp32 temporaries stay small (a 32k x 2048 embedding would otherwise need ~0.25 GiB per temporary)
                    rs, cs = torch.zeros_like(st["R"]), torch.zeros_like(st["C"])
                    for a in range(0, p.shape[0], self.CHUNK):
                        gb = p.grad[a:a + self.CHUNK].float()
                        rs[a:a + self.CHUNK] = (gb * gb).mean(1); cs += (gb * gb).sum(0)
                    st["R"].mul_(b2).add_(rs, alpha=1 - b2)
                    st["C"].mul_(b2).add_(cs / p.shape[0], alpha=1 - b2)
                    rmean = st["R"].mean().clamp_min(1e-30)
                    for a in range(0, p.shape[0], self.CHUNK):
                        sl = slice(a, a + self.CHUNK)
                        gb = p.grad[sl].float()
                        m = (st["mq"][sl].float() * st["ms"][sl]).mul_(b1).add_(gb, alpha=1 - b1)
                        sc = m.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
                        st["mq"][sl] = (m / sc + torch.rand_like(m)).floor_().clamp_(-127, 127).to(torch.int8)
                        st["ms"][sl] = sc
                        v = st["R"][sl, None] * st["C"][None, :] / rmean
                        upd = (m / bc1) / ((v / bc2).sqrt_().add_(eps))
                        p32 = p[sl].float()
                        if wd > 0:
                            p32.mul_(1 - lr * wd)
                        p32.add_(upd, alpha=-lr)
                        p[sl] = stochastic_round(p32, p.dtype)
                    continue
                g = p.grad.float()
                m = st["m"].mul_(b1).add_(g, alpha=1 - b1)
                v = st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                upd = (m / bc1) / ((v / bc2).sqrt_().add_(eps))
                p32 = p.float()
                if wd > 0:
                    p32.mul_(1 - lr * wd)
                p32.add_(upd, alpha=-lr)
                p.copy_(stochastic_round(p32, p.dtype))
        return loss
