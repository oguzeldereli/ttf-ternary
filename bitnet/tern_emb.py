"""--tern_emb tied|untied: a ternary token embedding (and output head), so only scales and norm gains stay float.

The embedding matrix E [vocab, dim] is W_ij = s0 * r_i * c_j * T_ij: trits T (packed 5 per byte, like the ternary
linear layers), learned per-row and per-column scales r, c (fp32, init 1) and a fixed base scale s0 that gives the float
embedding's init std (0.02). The lookup and the tied output head read the same W, computed once per forward (dense,
transient), so autograd sums both gradients into it. kernel mode: TTF trains T from dL/dT = dL/dW * s0 r c, called after
the backward (`tern_emb_step`; the linear layers' fused updates have run by then), with the same settings as the linear
layers. master mode: a float latent L, T = round(clip(L / mean|L|)) with the straight-through estimator (BitNet), trained
by the float optimizer. `untied`: a second matrix of the same kind as the output head (more trits, still ~1/10 of a bf16
embedding's bytes).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import pack_rows, unpack_rows

_DBG = bool(__import__('os').environ.get('TERN_EMB_DBG'))


class TernEmbed(nn.Module):
    def __init__(self, vocab: int, dim: int, mode: str = "kernel", std: float = 0.02):
        super().__init__()
        self.N, self.K, self.mode = vocab, dim, mode
        w = torch.randn(vocab, dim)
        t = (w / w.abs().mean()).round().clamp_(-1, 1).to(torch.int8)
        self.s0 = std / max(1e-6, t.float().pow(2).mean().sqrt().item())
        if mode == "master":
            self.latent = nn.Parameter(w)
        else:
            self.register_buffer("wpacked", pack_rows(t))
        self.row_scale = nn.Parameter(torch.ones(vocab))
        self.col_scale = nn.Parameter(torch.ones(dim))
        self.track = False
        self._W = None
        self._fresh = False

    def trits(self):
        if self.mode == "master":
            L = self.latent
            return (L / L.abs().mean().clamp_min(1e-8)).round().clamp(-1, 1)
        return unpack_rows(self.wpacked, self.K)

    def scale(self):
        return self.s0 * self.row_scale[:, None] * self.col_scale[None, :]

    def _compute(self):
        if self.mode == "master":
            L = self.latent
            Ln = L / L.abs().mean().clamp_min(1e-8)
            T = Ln + (Ln.round().clamp(-1, 1) - Ln).detach()          # straight-through estimator
        else:
            T = unpack_rows(self.wpacked, self.K).float()
        W = T * self.scale()
        if W.requires_grad:
            W.retain_grad()
        self._W = W
        return W

    @property
    def weight(self):
        if self._fresh:                     # the tied head of the forward that just did the lookup
            self._fresh = False
            return self._W
        return self._compute()

    def forward(self, idx):
        W = self._compute()
        self._fresh = True
        return F.embedding(idx, W)

    def grad_trits(self):
        """dL/dT (fp32, trit units) of the last forward, or None"""
        if self._W is None or self._W.grad is None:
            return None
        g = self._W.grad * self.scale().detach()
        self._W = None
        return g


def tern_emb_layers(model):
    return [m for m in model.modules() if isinstance(m, TernEmbed)]


def enable_tern_emb(model, kind: str, mode: str):
    """replace the float token embedding by a TernEmbed; `untied`: a second one as the output head"""
    V, d = model.tok_emb.weight.shape
    dev = model.tok_emb.weight.device
    model.tok_emb = TernEmbed(V, d, mode).to(dev)
    if kind == "untied":
        model.lm_head = TernEmbed(V, d, mode).to(dev)
    elif kind != "tied":
        raise SystemExit(f"--tern_emb {kind}: tied | untied")
    lays = tern_emb_layers(model)
    for i, m in enumerate(lays):
        m._ts_idx = 10_000 + i
    return sum(m.N * m.K for m in lays)


def tern_emb_step(model, state, c, ts_layer):
    """kernel mode, after the backward: the TTF update of each ternary embedding matrix"""
    for m in tern_emb_layers(model):
        g = m.grad_trits()
        if g is not None:
            before = m.wpacked.clone() if _DBG else None
            ts_layer(m, g, state, c)
            if _DBG:
                m._moves = int((unpack_rows(before, m.K) != unpack_rows(m.wpacked, m.K)).sum().item())
                print(f"tern_emb {m._ts_idx}: moves {m._moves}, |g| {g.abs().mean().item():.3e}", flush=True)
