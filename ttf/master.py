"""Standard BitNet b1.58 training: latent master weights + STE + AdamW.

The baseline TTF is measured against. The weight stored is a latent
high-precision matrix; the FORWARD PASS quantizes it to ternary (absmean, as in
BitNet b1.58) with a straight-through estimator, so the network that computes the
loss is ternary — only the accumulator behind it is not.

Latent dtype is fp32 by default: in bf16 (7 mantissa bits) an AdamW step of ~1e-4
on a weight of ~0.02 falls below half the representable gap and rounds away, and
the weight-decay term (lr*wd*w ~ 1e-6) always does. That silently cripples the
baseline, which is the one run that must not be crippled.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import BitTransformer


def _act_quant_ste(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token symmetric absmax activation quantization with straight-through."""
    qp = 2 ** (bits - 1) - 1
    scale = qp / x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    xq = (x * scale).round().clamp_(-qp - 1, qp) / scale
    return x + (xq - x).detach()


def _ternary_ste(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 absmean ternary {-1,0,1}*gamma with straight-through."""
    gamma = w.detach().abs().mean().clamp_min(1e-5)
    t = (w / gamma).round().clamp_(-1, 1) * gamma
    return w + (t - w).detach()


class MasterTernaryLinear(nn.Module):
    """Ternary-in-the-forward linear with a latent master weight (an nn.Parameter)."""

    def __init__(self, in_features: int, out_features: int, act_bits: int = 8,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.K, self.N, self.act_bits = in_features, out_features, act_bits
        w = torch.empty(out_features, in_features, dtype=dtype)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        self.weight = nn.Parameter(w)
        self.row_scale = self.col_scale = None   # --rc-scale in master mode (the float extras our runs carry)
        self.mag_A = None                        # --adapter-rank add:R in master mode

    def enable_rc_scale(self):
        dev = self.weight.device
        self.row_scale = nn.Parameter(torch.ones(self.N, device=dev))
        self.col_scale = nn.Parameter(torch.ones(self.K, device=dev))

    def enable_lowrank_mag(self, kind, r):
        """the same float adapter as the kernel layer's 'add' kind: W = ternary + A B^T (A starts at 0)"""
        assert kind == "add", "master mode: only the additive adapter"
        dev = self.weight.device
        self.mag_A = nn.Parameter(torch.zeros(self.N, r, device=dev))
        self.mag_B = nn.Parameter(torch.randn(self.K, r, device=dev) * (1.0 / self.K) ** 0.5)
        return [self.mag_A, self.mag_B]

    def _extras(self, x, y_fn):
        if self.col_scale is not None:
            x = x * self.col_scale.to(x.dtype)
        y = y_fn(x)
        if self.mag_A is not None:
            y = y + (x @ self.mag_B.to(x.dtype)) @ self.mag_A.to(x.dtype).T
        if self.row_scale is not None:
            y = y * self.row_scale.to(y.dtype)
        return y

    def forward(self, x):
        if self.row_scale is not None or self.mag_A is not None:
            return self._extras(x, self._forward)
        return self._forward(x)

    def _forward(self, x):
        # under autocast both operands go to bf16, matching the kernel path's math
        return F.linear(_act_quant_ste(x, self.act_bits), _ternary_ste(self.weight))

    def extra_repr(self):
        return f"in={self.K}, out={self.N}, ternary fwd (STE), latent {self.weight.dtype}"


def build_master_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                             dtype: torch.dtype = torch.float32):
    def make_linear(i, o):
        return MasterTernaryLinear(i, o, c.act_bits, dtype=dtype)
    return BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)


def split_params(model):
    """(latent master weights, embedding/head, norm gains) — for AdamW groups."""
    lays = [m for m in model.modules() if isinstance(m, MasterTernaryLinear)]
    master = [m.weight for m in lays]
    ids = {id(p) for p in master}
    emb, norms = [], []
    for n, p in model.named_parameters():
        if id(p) in ids:
            continue
        (norms if n.endswith("norm.weight") else emb).append(p)
    return master, emb, norms
