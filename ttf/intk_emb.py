"""--intk-emb E:H[:G]: Qwen's token embedding and output head as low-bit integer codes trained by TTF (no float master,
no Adam state for them): W = lo + s * q per group of G entries along each row, q in {0 .. 2^bits - 1} (E bits for the
embedding, H for the head; e.g. 2:3), lo and s per group (fp32, trained in the float tail with the norm gains). Started
from the round-to-nearest codes of the (converted, folded) float matrices.

TTF moves a code by one level when its accumulator reaches theta, as it moves a trit; the gradient it sees is dL/dq =
dL/dW * s (one level is one step of s). The dequantized matrix is formed once per optimizer step as an fp32 leaf and
reused by every micro-batch (the codes only change on the step's last micro-batch), so its gradient sums the step's
micro-batches; the update runs after the last one, from that sum (one call per matrix per step).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntKMatrix(nn.Module):
    def __init__(self, W, bits, g=64):
        super().__init__()
        N, K = W.shape
        assert K % g == 0, (K, g)
        self.N, self.K, self.g, self.bits, self.levels = N, K, g, bits, 2 ** bits - 1
        with torch.no_grad():
            Wf = W.float().reshape(N, K // g, g)
            lo = Wf.amin(-1); s = ((Wf.amax(-1) - lo) / self.levels).clamp_min(1e-12)
            q = ((Wf - lo[..., None]) / s[..., None]).round().clamp(0, self.levels)
        self.register_buffer("codes", q.reshape(N, K).to(torch.int8))
        self.lo, self.s = nn.Parameter(lo.contiguous()), nn.Parameter(s.contiguous())
        self.track, self._ts_host, self._W = False, False, None

    def dequant(self):
        q = self.codes.reshape(self.N, self.K // self.g, self.g).float()
        return (self.lo[..., None] + self.s[..., None] * q).reshape(self.N, self.K)

    def leaf(self):
        """the step's dequantized matrix (fp32): a leaf with its own gradient while training, else a plain tensor"""
        if not torch.is_grad_enabled():
            return self.dequant().detach()
        if self._W is None:
            self._W = self.dequant().detach().requires_grad_(True)
        return self._W

    @torch.no_grad()
    def step(self, ts_layer, state, c):
        """after the step's last backward: the group parameters' gradients and the TTF update of the codes"""
        W = self._W
        self._W = None
        if W is None or W.grad is None:
            return
        gW = W.grad.float().reshape(self.N, self.K // self.g, self.g)
        q = self.codes.reshape(self.N, self.K // self.g, self.g).float()
        for p, gp in ((self.lo, gW.sum(-1)), (self.s, (gW * q).sum(-1))):
            p.grad = gp if p.grad is None else p.grad + gp
        gq = (gW * self.s.detach()[..., None]).reshape(self.N, self.K)
        del gW, q
        ts_layer(self, gq, state, dict(c, micro=1))


class _Emb(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @property
    def weight(self):
        return self.m.leaf()

    def forward(self, idx):
        return F.embedding(idx, self.m.leaf()).to(torch.bfloat16)


class _Head(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @property
    def weight(self):
        return self.m.leaf()

    def forward(self, x):
        return F.linear(x, self.m.leaf().to(x.dtype))


def enable_intk_emb(model, spec, log=print):
    """replace hf.model.embed_tokens / hf.lm_head by int-k code matrices (spec 'E:H[:G]'); returns the two matrices"""
    p = [int(v) for v in spec.split(":")]
    eb, hb, g = p[0], p[1], (p[2] if len(p) > 2 else 64)
    hf = model.hf
    E, H = hf.model.embed_tokens.weight, hf.lm_head.weight
    assert E is not H, "--intk-emb: an untied head (ttfconv.fold unties it)"
    me, mh = IntKMatrix(E.data, eb, g).to(E.device), IntKMatrix(H.data, hb, g).to(H.device)
    for i, m in enumerate((me, mh)):
        m._ts_idx = 20_000 + i
    hf.model.embed_tokens = _Emb(me); hf.lm_head = _Head(mh)
    err = [float((m.dequant() - W.float()).norm() / W.float().norm()) for m, W in ((me, E.data), (mh, H.data))]
    log(f"--intk-emb {spec}: embedding {eb}-bit, head {hb}-bit codes in groups of {g} (rel err {err[0]:.4f} / "
        f"{err[1]:.4f}), trained by TTF; {(me.lo.numel() + me.s.numel() + mh.lo.numel() + mh.s.numel()) / 1e6:.1f}M "
        f"group parameters in the float tail")
    return me, mh


def intk_layers(model):
    return [m for m in model.modules() if isinstance(m, IntKMatrix)]
