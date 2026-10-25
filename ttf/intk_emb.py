"""--intk-emb E:H[:G]: Qwen's token embedding and output head as low-bit integer codes trained by TTF (no float master,
no Adam state for them): W = lo + s * q per group of G entries along each row, q in {0 .. 2^bits - 1} (E bits for the
embedding, H for the head; e.g. 2:3), lo and s per group (fp32, trained in the float tail with the norm gains). Started
from the round-to-nearest codes of the (converted, folded) float matrices.

TTF moves a code by one level when its accumulator reaches theta, as it moves a trit; the gradient it sees is dL/dq =
dL/dW * s (one level is one step of s). The dequantized matrix is formed once per optimizer step as a bf16 leaf and
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
        """the step's dequantized matrix (bf16): a leaf while training, else a plain tensor. Each micro-batch's gradient
        (bf16, rounded once) is moved by a post-accumulate hook into an fp32 sum (self._G), so the step's sum is fp32 and
        no bf16 .grad accumulates (at 27B: 2.5 GB per matrix instead of an fp32 leaf's 5 GB plus a bf16 copy)"""
        if not torch.is_grad_enabled():
            return self.dequant().detach().to(torch.bfloat16)
        if self._W is None:
            self._W = self.dequant().detach().to(torch.bfloat16).requires_grad_(True)
            self._G = None

            def _acc(p):
                g = p.grad.float()
                self._G = g if self._G is None else self._G.add_(g)
                p.grad = None
            self._W.register_post_accumulate_grad_hook(_acc)
        return self._W

    @torch.no_grad()
    def step(self, ts_layer, state, c):
        """after the step's last backward: the group parameters' gradients and the TTF update of the codes"""
        G_ = getattr(self, "_G", None)
        self._W = self._G = None
        if G_ is None:
            return
        import torch.distributed as dist
        if getattr(self, "dp_reduce", False) and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:   # data parallel: the codes
            dist.all_reduce(G_); G_.div_(dist.get_world_size())   # move on the mean gradient (the same on every rank)
        gW = G_.reshape(self.N, self.K // self.g, self.g)
        q = self.codes.reshape(self.N, self.K // self.g, self.g).float()
        for p, gp in ((self.lo, gW.sum(-1)), (self.s, (gW * q).sum(-1))):
            p.grad = gp if p.grad is None else p.grad + gp
        gq = (gW * self.s.detach()[..., None]).reshape(self.N, self.K)
        del gW, q, G_
        if self.N * self.K > CHUNKED_ABOVE[0]:
            ttf_rows(self, gq, state, c)
        else:
            ts_layer(self, gq, state, dict(c, micro=1))


CHUNKED_ABOVE = [400_000_000]   # entries: larger matrices (27B: 248320 x 5120) take ttf_rows (bounded temporaries)


@torch.no_grad()
def ttf_rows(l, g, state, c, rows=16384):
    """update.ts_layer's dense update (one call per step) for int-k codes in row chunks: the temporaries are
    [rows, K] instead of [N, K] (at 27B's 248320 x 5120 the full ones were ~30 GB). The same steps: v = R C / mean R
    (row / column EMAs of g^2, decay b2, bias-corrected), u = -lr g / (sqrt v + 1e-8), the rank-r accumulator A = U V^T
    (leak, the basis turned from V (U^T U) + u^T U), a code moves one level where |A| >= theta (within 0 .. levels),
    each move spends theta. V's first basis comes from a chunk-wise random projection (not ts_layer's single draw)."""
    from .update import _orth
    N, K = g.shape
    lr, theta, tau, b2 = c["lr_ratio"], c["theta"], c["tau"], c["b2"]
    st = state.setdefault("_ts", {}); key = id(l)
    t = st.get(("t", key), 0) + 1; st[("t", key)] = t
    R = torch.empty(N, device=g.device); Cs = torch.zeros(K, device=g.device)
    for i in range(0, N, rows):
        g2 = g[i:i + rows].square(); R[i:i + rows] = g2.mean(1); Cs += g2.sum(0); del g2
    C = Cs / N
    if ("v", key) in st:
        R0, C0 = st[("v", key)]; R = b2 * R0 + (1 - b2) * R; C = b2 * C0 + (1 - b2) * C
    else:
        R, C = (1 - b2) * R, (1 - b2) * C
    st[("v", key)] = (R, C)
    rs = (R / R.mean().clamp_min(1e-30) / (1 - b2 ** t)).sqrt(); cs = C.sqrt()   # sqrt(v_ij) = rs_i cs_j

    def u_of(i):
        return -g[i:i + rows] / (rs[i:i + rows, None] * cs[None, :] + 1e-8) * lr

    ra = min(int(getattr(l, "ts_rank", None) or c["r"]), N, K)
    if key not in state:
        gen = torch.Generator(device=g.device).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        X = torch.zeros(K, ra, device=g.device)
        for i in range(0, N, rows):
            u = u_of(i); X += u.T @ torch.randn(u.shape[0], ra, device=g.device, generator=gen); del u
        V = _orth(X); U = torch.empty(N, ra, device=g.device)
        for i in range(0, N, rows):
            U[i:i + rows] = u_of(i) @ V
    else:
        U, V = state[key]
        U = U * (1 - (lr ** float(c["tau_anneal"]) if c["tau_anneal"] else 1.0) / tau)
        X = V @ (U.T @ U)
        for i in range(0, N, rows):
            X += u_of(i).T @ U[i:i + rows]
        Vn = _orth(X); del X
        P = V.T @ Vn
        for i in range(0, N, rows):
            U[i:i + rows] = U[i:i + rows] @ P + u_of(i) @ Vn
        V = Vn
    nf = torch.zeros((), device=g.device, dtype=torch.long)
    for i in range(0, N, rows):
        A = U[i:i + rows] @ V.T; w0 = l.codes[i:i + rows]
        d = A.sign(); nw = w0.float() + d
        fire = (A.abs() >= theta) & (nw >= 0) & (nw <= l.levels); del A, nw
        D = torch.where(fire, d, torch.zeros_like(d)); del d
        w0.add_(D.to(torch.int8)); nf += fire.sum(); del fire
        U[i:i + rows] -= c["spend"] * (theta * (D @ V)); del D
    state[key] = (U, V)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + N * K


class _Emb(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @property
    def weight(self):
        return self.m.leaf()

    def forward(self, idx):
        return F.embedding(idx, self.m.leaf())


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
    err = [float((m.dequant() - W.float()).norm() / W.float().norm()) if not W.is_meta else float("nan")   # (--pipe:
           for m, W in ((me, E.data), (mh, H.data))]                                                         # another stage's)
    log(f"--intk-emb {spec}: embedding {eb}-bit, head {hb}-bit codes in groups of {g} (rel err {err[0]:.4f} / "
        f"{err[1]:.4f}), trained by TTF; {(me.lo.numel() + me.s.numel() + mh.lo.numel() + mh.s.numel()) / 1e6:.1f}M "
        f"group parameters in the float tail")
    return me, mh


def intk_layers(model):
    return [m for m in model.modules() if isinstance(m, IntKMatrix)]
