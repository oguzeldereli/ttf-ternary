"""--tern-emb tied|untied: a ternary token embedding (and output head), so only scales and norm gains stay float.

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
            T = unpack_rows(self.wpacked, self.K).to(getattr(self, "cdt", torch.float32))
        W = T * self.scale().to(T.dtype)
        if W.requires_grad:
            W.retain_grad()
        self._W = W
        return W

    @property
    def weight(self):
        if self._fresh and self._W is not None:   # the tied head of the forward that just did the lookup
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
        g = self._W.grad.float() * self.scale().detach()
        self._W = None
        self._fresh = False                 # an unread lookup's W (untied) is stale once the trits move
        return g


class TernEmbedWide(nn.Module):
    """--tern-emb tied_wide:M: a ternary embedding with the bytes of a float one. Packed trits cost 1.6 bits against 32
    for fp32, so a table of width M * dim (M = 20 matches fp32) holds the same memory; a ternary projection [dim, M dim]
    maps it to the model width. The effective embedding E = table @ proj^T [vocab, dim] is formed once per forward
    (dense, transient, bf16), used for the lookup and as the tied output head; both factors are TernEmbed matrices that
    TTF trains from their own dL/dT (the product's gradient, by autograd)."""

    def __init__(self, vocab: int, dim: int, mult: int, mode: str = "kernel"):
        super().__init__()
        D = mult * dim
        self.table = TernEmbed(vocab, D, mode, std=0.02)
        self.table.cdt = torch.bfloat16                           # the wide table expanded in bf16 (trits are exact)
        self.proj = TernEmbed(dim, D, mode, std=1.0 / D ** 0.5)   # E's entries keep the float embedding's std 0.02
        self.N, self.K = vocab, dim
        self._E = None
        self._fresh = False

    def _compute(self):
        E = (self.table._compute().to(torch.bfloat16) @ self.proj._compute().to(torch.bfloat16).T).float()
        self._E = E
        return E

    @property
    def weight(self):
        if self._fresh and self._E is not None:
            self._fresh = False
            return self._E
        return self._compute()

    def forward(self, idx):
        E = self._compute()
        self._fresh = True
        return F.embedding(idx, E)


def tern_emb_layers(model):
    return [m for m in model.modules() if isinstance(m, TernEmbed)]


def enable_tern_emb(model, kind: str, mode: str):
    """replace the float token embedding by a TernEmbed; `untied`: a second one as the output head"""
    V, d = model.tok_emb.weight.shape
    dev = model.tok_emb.weight.device
    if kind.startswith("tied_wide"):                 # tied_wide[:M], M = 20 by default (the bytes of an fp32 table)
        mult = int(kind.split(":")[1]) if ":" in kind else 20
        assert model.lm_head is None, "--tern-emb tied_wide needs a tied model"
        model.tok_emb = TernEmbedWide(V, d, mult, mode).to(dev)
        lays = tern_emb_layers(model)
        for i, m in enumerate(lays):
            m._ts_idx = 10_000 + i
        return sum(m.N * m.K for m in lays)
    model.tok_emb = TernEmbed(V, d, mode).to(dev)
    if kind == "untied":
        model.lm_head = TernEmbed(V, d, mode).to(dev)
    elif kind != "tied":
        raise SystemExit(f"--tern-emb {kind}: tied | untied")
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
