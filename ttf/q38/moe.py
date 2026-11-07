"""moe_ttf.py: ternary mixture-of-experts layers and their TTF update, batched over the experts (P, 5 Nov 2026, for S's
Qwen3.8-Flash-Next conversion: 48 layers x 512 experts, gate/up fused [1280, 2560] and down [2560, 640] per expert).

User (5 Nov ~01:30): "you can use ternary adapters of equal size to get rid of that 40GB + optimize state, put
accumulator at rank 0.3 into host rama and stream it".

TernaryExperts(E, K, N): E experts' trits packed [E, N, Kp] (5 per byte along K, as KernelTernaryLinear), a fixed
beta [E], learned row [E, N] / column [E, K] scales, an optional fixed input mean mu [E, K] (centring) with a learned
bias [E, N], and an optional ternary low-rank adapter (TA [E, N, ra], TB [E, ra, K] trits, per-rank scale cb [E, ra],
per-expert scale sa [E]). One expert's output is the core layer's:
    y = r * (beta T q(c (x - mu)) + sa TA (cb * (TB c (x - mu)))) + bias
with q the per-token int8 activation quantisation (straight-through in the backward). The input arrives sorted by
expert, x [T, K] with offsets [E + 1] (the router sort is the caller's).

ExpertTTF: the TTF update of all experts of one TernaryExperts at once, the core's one-batch factored rule
(update._ts_layer_factored with --accum 1: v from the step's sketched g^2, the exact basis turn, the firing and the
spend), applied to every expert that saw tokens this step, from the step's factors (gy, x) collected over all its
micro-batches (the layer-major trainer runs the whole step through one layer at a time, so the factors of the whole
step are there at once). Experts with no tokens are left as they are (as a core layer without a gradient). The state
(U [E, N, r], V [E, K, r], R [E, N], C [E, K], t [E]) lives wherever the caller keeps it (host, pinned); the update
takes it on the device and gives it back.

Equivalence: test_moe_ttf.py checks the forward / input gradient / factors against per-expert KernelTernaryLinear
layers and the update against the core _ts_layer_factored run per expert (seeds="core": the core's own sketch and init
seeds per expert), on CPU.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from ttf.kernel import pack_rows, unpack_rows, act_quant_i8
try:
    from ttf.kernel import act_quant_i8_fast      # (one kernel per row, bit-identical codes and scales)
except ImportError:
    act_quant_i8_fast = None
from ttf.update import _orth
from ttf.kernel import ttf_fire
try:
    from ttf.q38 import moe_gg as GGK
except ImportError:      # (no Triton: the padded path)
    GGK = None


def pack3(T):
    """int8 trits [E, N, K] -> uint8 [E, N, ceil(K / 5)]"""
    E, N, K = T.shape
    return pack_rows(T.reshape(E * N, K).contiguous()).view(E, N, -1)


def _beta_nnz(T):
    """the core's beta for trits T [N, K]: (nnz / N)^-1/2, clamped at 1 (kernel.beta_from_nnz)"""
    return (T.ne(0).sum().float() / T.shape[0]).clamp_min(1.0).rsqrt()


def grouped_to_dict(g):
    o, gy, x = g
    ol = o.tolist()
    return {e: (gy[ol[e]:ol[e + 1]], x[ol[e]:ol[e + 1]]) for e in range(len(ol) - 1) if ol[e + 1] > ol[e]}


def unpack3(P, K):
    """uint8 [E, N, Kp] -> int8 [E, N, K]"""
    E, N, Kp = P.shape
    return unpack_rows(P.reshape(E * N, Kp).contiguous(), K).view(E, N, K)


def _bmmb(A, B):
    """A (bf16 factor) @ B with B rounded to bf16, products accumulated in fp32, fp32 out (as update._mmb, batched)"""
    if A.dtype != torch.bfloat16:
        return A @ B
    Bb = B.to(torch.bfloat16)
    if A.is_cuda:
        try:
            return torch.bmm(A, Bb, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            pass
    return A.float() @ Bb.float()


def _seg(offs):
    o = offs.tolist() if torch.is_tensor(offs) else list(offs)
    return [(e, o[e], o[e + 1]) for e in range(len(o) - 1) if o[e + 1] > o[e]]


def _offs_t(offs, dev):
    return offs.to(dev, torch.long) if torch.is_tensor(offs) else torch.tensor(list(offs), dtype=torch.long, device=dev)


def _ev(v, pd):
    """a per-expert tensor v [E, ...] on the expert-sorted rows: E = 1 a broadcast [1, ...] (its gradient a plain sum);
    moe_gg rows with a gradient: EVec (segment sums, no sorted index_put); else v[e_of]"""
    if v.shape[0] == 1:
        return v[:1]
    if isinstance(pd, _Grp) and v.requires_grad and torch.is_grad_enabled():
        return GGK.EVec.apply(v, pd.e_of, pd.sch)
    return v[pd.e_of]


class _Pad:
    """the expert-sorted rows [T, D] <-> padded [E, C, D] (C = the largest expert's count): one index for both ways,
    so each per-expert product is one batched GEMM instead of a Python loop over the experts"""
    def __init__(self, offs, E, dev):
        o = _offs_t(offs, dev)
        cnt = o[1:] - o[:-1]
        self.T = int(o[-1]); self.C = max(1, int(cnt.max())) if E > 0 else 1; self.E = E
        self.e_of = torch.repeat_interleave(torch.arange(E, device=dev), cnt, output_size=self.T)
        self.flat = self.e_of * self.C + torch.arange(self.T, device=dev) - o[:-1][self.e_of]

    def pad(self, x):
        return x.new_zeros(self.E * self.C, x.shape[-1]).index_copy(0, self.flat, x).view(self.E, self.C, -1)

    def unpad(self, y):
        return y.reshape(self.E * self.C, -1).index_select(0, self.flat)


GROUPED = [True]    # CUDA: moe_gg's Triton grouped GEMMs over the expert segments (no padding to the hottest expert, no
                    # host sync; torch._grouped_mm on sm_120 is a host loop over the groups). Forward int8 x trits with
                    # exact int32 sums (as the padded path); backward and the adapter bf16 with fp32 accumulation


FUSED_DENSE = [True]   # E = 1 through moe_gg's Pre / Post (fp32, one rounding; False: the eager bf16 ops)
SPLIT = 256
_SPL = {}


def _split_sched(T, dev):
    """T rows as ceil(T / SPLIT) segments of SPLIT rows (moe_gg Sched; cached per T)"""
    k = (T, str(dev))
    if k not in _SPL:
        o = torch.arange(0, T + SPLIT, SPLIT, device=dev).clamp_max(T)
        if o.numel() > 1 and int(o[-2]) == T:
            o = o[:-1]
        _SPL[k] = GGK.Sched(o, T)
    return _SPL[k]


_G1 = {}


def _grp1(T, dev):
    k = (T, str(dev))
    if k not in _G1:
        _G1[k] = _Grp([0, T], 1, dev, T)
    return _G1[k]


def _h2d(t, dev):
    """a small host tensor to the device without a sync (pinned staging, non_blocking)"""
    if torch.device(dev).type != "cuda" or t.is_cuda:
        return t.to(dev)
    return t.pin_memory().to(dev, non_blocking=True)


class _Grp:
    """the expert segments for moe_gg: its row-tile schedule and each row's expert"""
    def __init__(self, offs, E, dev, T):
        o = _offs_t(offs, dev)
        self.e_of = torch.repeat_interleave(torch.arange(E, device=dev), o[1:] - o[:-1], output_size=T)
        self.sch = GGK.Sched(o, T)




FIRE_B = ["kernel"]  # update_b's firing: "kernel" (the core's ttf_fire, batched over the chunk: moe_gg.bfire; fp32
                     # decisions from a bf16 bound + exact fix, spend D V ~fp32; on CUDA) or "torch" (A = U V^T in fp32)
ORTH_B = ["qr"]   # update_b's orthonormal bases: "qr" (Householder, cuSOLVER; one matrix at a time, ~2-10 ms each at
                  # the expert shapes whatever the batch) or "rchol" (spanqr's rchol3_ns3 recipe batched: GEMMs, batched
                  # r x r Cholesky and triangular solves; 528/528 on spanqr's real TTF robustness cases, where plain
                  # cholqr2 failed 79)
_ZB = {}


def _p2(m):
    """2^-round(log2 m) (exact scaling), 1 where m is 0 (spanqr rcholqr._pow2_scale)"""
    return torch.where(m > 0, torch.exp2(-torch.round(torch.log2(m.clamp_min(1e-38)))), torch.ones_like(m))


RCHOL_FALLBACK = [0]   # the matrices _rchol_b handed to Householder QR (a Cholesky broke down), a running count
_DEFER = [False]       # update_b: _rchol_b leaves its broken-matrix mask in _RB_LAST instead of a host check
_RB_LAST = [None]
_FORCE_HH = [False]    # update_b's repair pass: Householder QR for every matrix
DEFER_LAYER = [False]  # the trainer: update_b leaves its broken-expert masks in _PEND; finish_repairs checks them at once
_PEND = []


def finish_repairs():
    """one host check for every update_b since the last call (a layer's modules); repairs the broken experts (see
    ExpertTTF._repair). Returns [(mod, moves)] of the repaired modules. Call before the states are written back."""
    if not _PEND:
        return []
    fl = torch.stack([p[2].any() for p in _PEND]).cpu().tolist()
    out = [(p[1], p[0]._repair(*p[1:])) for f, p in zip(fl, _PEND) if f]
    _PEND.clear()
    return out


def _rchol_b(X, eps=2e-7, alphas=(1e-6, 1e-6, 1e-7), ns=3):
    """spanqr rcholqr(X, ns=3) on a batch [B, K, r]: per matrix the exact power-of-2 scalings, the seeded unit-column
    completion eps Z (rank-deficient or zero columns get arbitrary orthonormal directions), three shifted Cholesky
    passes applied by triangular solves (the shift alpha ||G||_F keeps every Cholesky from breaking down and does not
    change the span), then Newton-Schulz polish. Same Z as spanqr for a (K, r) shape (seed 20261018)."""
    B, K, r = X.shape
    key = (K, r, str(X.device))
    if key not in _ZB:
        g = torch.Generator(device=X.device).manual_seed(20261018)
        Z = torch.randn(K, r, device=X.device, generator=g)
        _ZB[key] = Z / Z.norm(dim=0, keepdim=True)
    Y = X * _p2(X.abs().amax(dim=(1, 2), keepdim=True))
    Y = Y * _p2(Y.norm(dim=1, keepdim=True).amax(dim=2, keepdim=True))
    Y = Y + eps * _ZB[key]
    I = torch.eye(r, device=X.device).expand(B, r, r)
    bad = None
    for a in alphas:
        G = Y.transpose(1, 2) @ Y
        G.diagonal(dim1=1, dim2=2).add_(a * torch.linalg.matrix_norm(G)[:, None] + 1e-30)
        R, info = torch.linalg.cholesky_ex(G, upper=True)
        bad = info > 0 if bad is None else bad | (info > 0)
        # Y R^-1 as a GEMM with the small [r, r] inverse (the batched right trsm over [K, r] is ~4x slower at the
        # expert shapes; the same orthogonality and span, 5 Nov: scratchpad rchol_var.py)
        Y = Y @ torch.linalg.solve_triangular(R, I, upper=True)
    for _ in range(ns):
        Y = torch.baddbmm(Y, Y, Y.transpose(1, 2) @ Y, beta=1.5, alpha=-0.5)    # 1.5 Y - 0.5 Y (Y^T Y), one pass
    # a Cholesky that broke down: Householder QR for those matrices only (as the core's cholqr2 check; one host sync
    # per call). It happens: the fp32 Gram of a near-rank-1 Y (an expert with 1-2 tokens, K 2560, r 384) carries
    # rounding of ~3e-6 ||G|| in norm, above the first shift 1e-6 ||G||, so G + shift is indefinite (6 Nov: such an
    # expert's U / V went NaN for good, 1-13 experts per layer in run2's ckpt@50; scratchpad nanrep4.py)
    bad = bad | ~torch.isfinite(Y).flatten(1).all(1)
    if _DEFER[0]:          # (update_b: no sync here; the caller collects the mask and repairs once per module)
        _RB_LAST[0] = bad
        return Y.contiguous()
    if bool(bad.any()):
        ib_ = bad.nonzero().flatten()
        Y[ib_] = torch.linalg.qr(X[ib_])[0]
        RCHOL_FALLBACK[0] += int(ib_.numel())
    return Y.contiguous()


_NANDUMP = __import__("os").environ.get("NANDUMP")      # (debug: the first expert whose U / V turn non-finite)
_NANDUMP_DONE = []


def _nan_dump(mod, st, idx, t, gy, x, ia, ib, R, Cc, R0, C0, U, V, lr, lk, th, ini):
    bad = (~torch.isfinite(U)).flatten(1).any(1) | (~torch.isfinite(V)).flatten(1).any(1)
    if not bool(bad.any()):
        return
    j = int(bad.nonzero()[0, 0])
    e = int(idx[j])
    U0, V0 = st["U"][e].float(), st["V"][e].float()
    d = dict(e=e, t=int(t[j]), ini=list(ini), lr=lr, lk=lk, gy=gy[j].cpu(), x=x[j].cpu(), ia=ia[j].cpu(), ib=ib[j].cpu(),
             R=R[j].cpu(), C=Cc[j].cpu(), R0=R0[j].cpu(), C0=C0[j].cpu(), U0=U0.cpu(), V0=V0.cpu(), U=U[j].cpu(), V=V[j].cpu(),
             nbad=int(bad.sum()), N=mod.N, K=mod.K)
    torch.save(d, _NANDUMP)
    print("NANDUMP saved", e, "t", d["t"], "nbad", d["nbad"], "U0 finite", bool(torch.isfinite(U0).all()),
          "gy finite", bool(torch.isfinite(d["gy"]).all()), "x finite", bool(torch.isfinite(d["x"]).all()), flush=True)
    _NANDUMP_DONE.append(1)


DIAG = [False]      # update_b diagnostics into mod._diag (the 5 Nov runaway study): gradient spike, update size, orth


def _dg(mod, k, v, how="max"):
    d = mod.__dict__.setdefault("_diag", {})
    if how == "max":
        d[k] = max(d.get(k, float("-inf")), v)
    else:
        s_, n_ = d.get(k + "_acc", (0.0, 0))
        d[k + "_acc"] = (s_ + v, n_ + 1); d[k] = (s_ + v) / (n_ + 1)


TF32_TURN = [False]   # the core's --tf32: the basis turn's fp32 products on TF32; the orthonormalisation in full fp32


class _tf32:
    """the fp32 matmuls on TF32 (on) or full fp32 inside the block, restored after. The new fp32_precision API only
    (torch 2.13 raises on a mix with the legacy allow_tf32): the flag is process-wide, so only the update (one thread)
    sets it; the trainer's other threads run no fp32 cuBLAS products"""
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        m = torch.backends.cuda.matmul
        self.prev = m.fp32_precision
        m.fp32_precision = "tf32" if self.on else "ieee"

    def __exit__(self, *a):
        torch.backends.cuda.matmul.fp32_precision = self.prev


def _qr_b(X):
    """orthonormal basis of each [K, r] in the batch (only the span matters to TTF); always full fp32 (as update._orth)"""
    with _tf32(False):
        _RB_LAST[0] = None
        if (ORTH_B[0] == "rchol" and not _FORCE_HH[0] and X.is_cuda and X.dtype == torch.float32
                and X.shape[1] >= X.shape[2]):
            return _rchol_b(X)
        return torch.linalg.qr(X)[0]


def _bmm32(A, B):
    """A @ B, fp32 out (CUDA: bf16 operands, fp32 accumulation, as torch.mm(out_dtype=fp32))"""
    if A.is_cuda:
        try:
            return torch.bmm(A, B, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            return torch.bmm(A.float(), B.float())
    return torch.bmm(A, B)


class _ExpertGemm(torch.autograd.Function):
    """y [T, N] = beta_e (q(x) T_e^T) per expert; backward: gx = (beta_e gy) T_e (straight through q), and with capture
    on the factors (beta_e gy, q(x)) of the micro-batch go to mod.collect (no weight gradient formed). One padded bmm
    over the experts per product."""
    @staticmethod
    def forward(ctx, x, mod, pd):
        aq = act_quant_i8_fast if (x.is_cuda and act_quant_i8_fast is not None) else act_quant_i8
        xq, xs = aq(x.reshape(-1, mod.K), mod.act_bits)
        # y = (codes @ T^T) * xs * beta: the codes (|q| <= 128) and trits are exact in bf16 and their products sum
        # exactly in fp32 (K * 128 < 2^24), as the core's int32 GEMM; then the scales in fp32, one rounding (the core's
        # tern_gemm_i8 returns bf16)
        cdt = torch.bfloat16 if x.is_cuda else torch.float32
        W = None if isinstance(pd, _Grp) else mod.w_float(cdt)  # [E, N, K] (exact: trits)
        if isinstance(pd, _Grp):      # (exact int32 sums, then * xs * beta in fp32 and one rounding, in the epilogue)
            y = GGK.gg(xq, mod.w_float(torch.int8), True, pd.sch, rs=xs, es=mod.beta,
                       out_dtype=torch.bfloat16).to(x.dtype)
        else:
            acc = pd.unpad(_bmm32(pd.pad(xq.to(cdt)), W.transpose(1, 2)))      # [T, N] fp32 (exact sums)
            y = acc * xs[:, None] * _ev(mod.beta, pd)[:, None]
            y = (y.to(torch.bfloat16) if x.is_cuda else y).to(x.dtype)
        if mod.capture:      # (the factor x, the dequantised int8 input, formed in the backward)
            ctx.save_for_backward(xq, xs)
        ctx.mod, ctx.pd, ctx.xdt = mod, pd, x.dtype
        return y

    @staticmethod
    def backward(ctx, gy):
        mod, pd = ctx.mod, ctx.pd
        cdt = torch.bfloat16 if gy.is_cuda else gy.dtype       # (the core's tern_gemm_dx: bf16 gy, bf16 cuBLAS)
        gyb = gy * _ev(mod.beta, pd)[:, None].to(gy.dtype)
        if isinstance(pd, _Grp):
            gx = GGK.gg(gyb.to(cdt).contiguous(), mod.w_float(torch.int8), False, pd.sch,
                        out_dtype=cdt).to(gy.dtype)  # (bf16 out, as cuBLAS: the fp32 sums rounded once in the epilogue)
        else:
            W = mod.w_float(cdt)
            gx = pd.unpad(torch.bmm(pd.pad(gyb.to(cdt)), W)).to(gy.dtype)
        if mod.capture:
            xq, xs = ctx.saved_tensors
            if xq.is_cuda and ctx.xdt == torch.bfloat16 and GGK is not None:   # (the codes kept, dequantised in the take)
                mod.collect(pd, gyb, (xq, xs.to(ctx.xdt)))
            else:
                mod.collect(pd, gyb, xq.to(ctx.xdt) * xs[:, None].to(ctx.xdt))
        return gx, None, None


class TernaryExperts(nn.Module):
    def __init__(self, E, K, N, act_bits=8, centring=False, adapter_rank=0, device=None):
        super().__init__()
        self.E, self.K, self.N, self.act_bits = E, K, N, act_bits
        dev = device
        self.register_buffer("wpacked", torch.zeros(E, N, (K + 4) // 5, dtype=torch.uint8, device=dev))
        self.wpacked.fill_(pack_rows(torch.zeros(1, K, dtype=torch.int8))[0, 0])   # (all-zero trits)
        self.register_buffer("beta", torch.ones(E, device=dev))
        self.beta_nnz = True    # beta follows the trits (core use_beta); False: beta fixed at its loaded value
        self.row_scale = nn.Parameter(torch.ones(E, N, device=dev))
        self.col_scale = nn.Parameter(torch.ones(E, K, device=dev))
        if centring:
            self.register_buffer("in_mean", torch.zeros(E, K, device=dev))
            self.out_bias = nn.Parameter(torch.zeros(E, N, device=dev))
        else:
            self.in_mean, self.out_bias = None, None
        self.ra = adapter_rank
        if adapter_rank:
            z = pack_rows(torch.zeros(1, adapter_rank, dtype=torch.int8))[0, 0]
            self.register_buffer("ta_packed", torch.full((E, N, (adapter_rank + 4) // 5), int(z), dtype=torch.uint8,
                                                         device=dev))
            self.register_buffer("tb_packed", pack3(torch.randint(-1, 2, (E, adapter_rank, K), dtype=torch.int8,
                                                                  device=dev)))
            self.ad_cb = nn.Parameter(torch.full((E, adapter_rank), 1.0 / math.sqrt(K), device=dev))
            self.ad_sa = nn.Parameter(torch.full((E,), 1.0 / math.sqrt(adapter_rank), device=dev))
        self.capture = False
        self._fac = []          # the step's factors: [(segs, gy_beta [T, N], xf [T, K])] per micro-batch
        self._wc = None
        self._nnz = None

    # --- conversion ---
    @torch.no_grad()
    def load_expert(self, e, T, beta, r, c, mu=None, bias=None):
        """one expert from the conversion: trits int8 [N, K], beta (scalar), row [N], column [K] scales, mu [K], bias [N]"""
        T = T.to(torch.int8).to(self.wpacked.device)
        self.wpacked[e] = pack_rows(T)
        self._nnz = None
        # beta None: the core's beta_from_nnz, (nnz / N)^-1/2 clamped at 1, kept current after each firing (beta_nnz)
        self.beta[e] = _beta_nnz(T) if beta is None else float(beta)
        self.row_scale[e] = r.to(self.row_scale.device).float()
        self.col_scale[e] = c.to(self.col_scale.device).float()
        if mu is not None:
            self.in_mean[e] = mu.to(self.in_mean.device).float()
        if bias is not None:
            self.out_bias[e] = bias.to(self.out_bias.device).float()
        self._wc = None
        self._nnz = None

    def w_float(self, dtype):
        """the trits as [E, N, K] in dtype, unpacked once per change of the trits (invalidate())"""
        k = (self.wpacked.data_ptr(), self.wpacked._version, dtype)
        if self._wc is None or self._wc[0] != k:
            self._wc = (k, unpack3(self.wpacked, self.K).to(dtype))
        return self._wc[1]

    def invalidate(self):
        self._wc = None

    def collect(self, pd, gyb, xf):
        """xf the input factor [T, K], or (codes int8 [T, K], scales bf16 [T]): x = bf16(codes * scales), formed in
        take_grouped after the gather (the same values; half the bytes moved)"""
        xf = tuple(v.detach() for v in xf) if isinstance(xf, tuple) else xf.detach()
        self._fac.append((pd.e_of, gyb.detach(), xf))

    def take_grouped(self):
        """the step's factors grouped by expert over all micro-batches: (offsets [E + 1], gy [T, N], x [T, K]), rows of
        expert e at offsets[e]:offsets[e + 1] (in micro-batch order); one stable sort, no per-expert work"""
        fac, self._fac = self._fac, []
        if not fac:
            return None
        e = torch.cat([f[0] for f in fac]); gy = torch.cat([f[1] for f in fac])
        order = torch.argsort(e, stable=True)
        o = torch.zeros(self.E + 1, dtype=torch.long, device=e.device)
        o[1:] = torch.cumsum(torch.zeros(self.E, dtype=torch.long, device=e.device).scatter_add_(
            0, e.long(), torch.ones_like(e, dtype=torch.long)), 0)          # (the counts; bincount syncs on CUDA)
        if isinstance(fac[0][2], tuple):
            q = torch.cat([f[2][0] for f in fac]); sc = torch.cat([f[2][1] for f in fac])
            return o, gy[order], GGK.gather_dq(q, sc, order)
        x = torch.cat([f[2] for f in fac])
        return o, gy[order], x[order]

    def take_factors(self):
        """take_grouped as {e: (gy [T_e, N], x [T_e, K])} (tests and the per-expert update)"""
        g = self.take_grouped()
        return {} if g is None else grouped_to_dict(g)

    def _expert_vec(self, v, offs, n):
        """a per-expert vector [E, D] expanded to the sorted tokens [n, D]"""
        cnt = (offs[1:] - offs[:-1]) if torch.is_tensor(offs) else torch.tensor(
            [offs[i + 1] - offs[i] for i in range(len(offs) - 1)])
        idx = torch.repeat_interleave(torch.arange(self.E, device=v.device), cnt.to(v.device), output_size=n)
        return v[idx]

    def forward(self, x, offs):
        if x.is_cuda and GROUPED[0] and GGK is not None:
            # (E = 1, offsets [0, T] from the host: one cached _Grp per T, no host -> device copy per call)
            pd = _grp1(x.shape[0], x.device) if self.E == 1 and not torch.is_tensor(offs) else \
                _Grp(offs, self.E, x.device, x.shape[0])
        else:
            pd = _Pad(offs, self.E, x.device)
        if isinstance(pd, _Grp) and (self.E > 1 or FUSED_DENSE[0]):
            # moe_gg's fused row scales: no [T, D] expansion of the per-expert vectors, fp32 arithmetic, one rounding.
            # E = 1 (the dense projections): the rows split into SPLIT-row segments, each a copy of the one expert's
            # vectors (expand: autograd sums the segments' gradients), so the kernels' column sums run in parallel
            sch, cs, mu, rs, ob, sa = pd.sch, self.col_scale, self.in_mean, self.row_scale, self.out_bias, \
                (self.ad_sa if self.ra else None)
            if self.E == 1:
                sch = _split_sched(x.shape[0], x.device)
                S = sch.E
                cs, rs = cs.expand(S, -1), rs.expand(S, -1)
                mu = mu.expand(S, -1) if mu is not None else None
                ob = ob.expand(S, -1) if ob is not None else None
                sa = sa.expand(S) if sa is not None else None
            x = GGK.Pre.apply(x, cs, mu, sch)
            y = _ExpertGemm.apply(x, self, pd)
            o = self._adapter(x, pd, raw=True) if self.ra else None
            return GGK.Post.apply(y, o, sa, rs, ob, sch)
        if self.in_mean is not None:
            x = x - _ev(self.in_mean, pd).to(x.dtype)
        x = x * _ev(self.col_scale, pd).to(x.dtype)
        y = _ExpertGemm.apply(x, self, pd)
        if self.ra:
            y = y + self._adapter(x, pd)
        y = y * _ev(self.row_scale, pd).to(y.dtype)
        if self.out_bias is not None:
            y = y + _ev(self.out_bias, pd).to(y.dtype)
        return y

    def _adapter(self, x, pd, raw=False):
        if self.capture and torch.is_grad_enabled():
            # the adapter's trits as float leaves, one pair per step (their .grad sums over the micro-batches; the
            # dense adapter gradient for AdapterTTF, [E, N, ra] + [E, ra, K]: small at ra 16)
            if getattr(self, "_ad_leaf", None) is None:
                self._ad_leaf = (unpack3(self.ta_packed, self.ra).to(x.dtype).requires_grad_(True),
                                 unpack3(self.tb_packed, self.K).to(x.dtype).requires_grad_(True))
            TA, TB = self._ad_leaf
        else:
            TA = unpack3(self.ta_packed, self.ra).to(x.dtype)      # [E, N, ra]
            TB = unpack3(self.tb_packed, self.K).to(x.dtype)       # [E, ra, K]
        if isinstance(pd, _Grp):
            # (the scales in fp32: their gradients are sums over all rows, which bf16 products would blur)
            xb = x.to(torch.bfloat16)
            z = GGK.GG.apply(xb, TB.to(torch.bfloat16), True, pd.sch) * _ev(self.ad_cb, pd).float()
            o = GGK.GG.apply(z.to(torch.bfloat16), TA.to(torch.bfloat16), True, pd.sch)
            if raw:      # (fp32, before ad_sa: Post applies it)
                return o
            return (o * _ev(self.ad_sa, pd)[:, None].float()).to(x.dtype)
        z = torch.bmm(pd.pad(x), TB.transpose(1, 2)) * self.ad_cb[:, None, :].to(x.dtype)   # [E, C, ra]
        return pd.unpad(torch.bmm(z, TA.transpose(1, 2)) * self.ad_sa[:, None, None].to(x.dtype))


class TernLinear(TernaryExperts):
    """a dense ternary projection as one expert (E = 1): nn.Linear's call, forward(x [..., K]) -> [..., N], the same
    buffers / parameters (and state-dict keys) as TernaryExperts, so the conversion files and the trainer treat the
    dense projections and the routed experts alike"""
    def __init__(self, K, N, **kw):
        super().__init__(1, K, N, **kw)

    def forward(self, x):
        sh = x.shape
        x2 = x.reshape(-1, self.K)
        return super().forward(x2, [0, x2.shape[0]]).view(*sh[:-1], self.N)


class AdapterTTF:
    """TTF for the ternary adapter's trits (TA [E, N, ra], TB [E, ra, K]) from their dense gradients: a full second
    moment v (the tensors are small), u = -lr g / sqrt(v / (1 - b2^t)), A <- (1 - 1 / tau) A + u, a move where
    |A| >= theta and the trit stays in {-1, 0, 1}, spend theta. Experts with no tokens this step (zero gradient) are
    left as they are. The state per module: {"A_a", "v_a", "A_b", "v_b" fp32, "t" [E]}."""
    @staticmethod
    def new_state(E, N, K, ra, device="cpu"):
        z = lambda *s_: torch.zeros(*s_, device=device)
        return {"A_a": z(E, N, ra), "v_a": z(E, N, ra), "A_b": z(E, ra, K), "v_b": z(E, ra, K),
                "t": torch.zeros(E, dtype=torch.int64, device=device)}

    @torch.no_grad()
    def update(self, mod, st, c, ret_t=False):
        lf = getattr(mod, "_ad_leaf", None)
        mod._ad_leaf = None
        if lf is None or lf[0].grad is None:
            return 0
        ga, gb = lf[0].grad.float(), lf[1].grad.float()
        act = (ga.abs().sum((1, 2)) + gb.abs().sum((1, 2))) > 0                   # [E]
        lr, tau, b2, sp, th = c["lr_ratio"], c["tau"], c["b2"], c["spend"], float(c["theta"])
        st["t"] += act.to(st["t"].dtype)
        bc = (1 - b2 ** st["t"].clamp_min(1).double()).float()[:, None, None]
        nf = 0
        for key, g, P, K_ in (("a", ga, mod.ta_packed, mod.ra), ("b", gb, mod.tb_packed, mod.K)):
            m_ = act[:, None, None]
            v = torch.where(m_, b2 * st["v_" + key] + (1 - b2) * g * g, st["v_" + key])
            u = -lr * g / (v / bc).sqrt().clamp_min(1e-30)
            A = torch.where(m_, (1 - 1 / tau) * st["A_" + key] + torch.where(v > 0, u, torch.zeros_like(u)), st["A_" + key])
            w0 = unpack3(P, K_)
            d = A.sign()
            fire = (A.abs() >= th) & ((w0.float() + d).abs() <= 1) & m_
            D = torch.where(fire, d, torch.zeros_like(d))
            nf = nf + fire.sum()
            P.copy_(pack3(w0 + D.to(torch.int8)))
            st["A_" + key] = A - sp * th * D; st["v_" + key] = v
        return nf if ret_t or not torch.is_tensor(nf) else int(nf)


class ExpertTTF:
    """the core one-batch factored TTF update for all experts of TernaryExperts layers. c: the core's ts_cfg keys
    (lr_ratio, theta, tau, b2, spend, factored = the sketch size k) plus r (accumulator rank). theta may be a [E] tensor
    (per-expert thresholds). seeds "core": each expert's sketches and initial basis from the core's seeds with
    _ts_idx = idx0 + e (equivalence tests); "layer": one sketch pair per layer and step, shared by its experts (cheaper;
    E[S S^T] = I per expert all the same)."""
    def __init__(self, seeds="layer", chunk=32):
        self.seeds, self.chunk = seeds, chunk

    @staticmethod
    def new_state(E, N, K, r, device="cpu", dtype=torch.float32, pin=False):
        ra = min(r, N, K)
        # dtype: the storage of U and V (the core's --ttf-store: fp32 or bf16; read as fp32 in the update, written
        # back rounded); R and C stay fp32
        z = lambda *s, dt=dtype: (torch.zeros(*s, dtype=dt, pin_memory=True) if pin
                                  else torch.zeros(*s, dtype=dt, device=device))
        return {"U": z(E, N, ra), "V": z(E, K, ra), "R": z(E, N, dt=torch.float32), "C": z(E, K, dt=torch.float32),
                "t": torch.zeros(E, dtype=torch.int64), "ra": ra}

    @torch.no_grad()
    def update(self, mod, fac, st, c, idx0=0, step_seed=0):
        """mod: TernaryExperts (trits changed in place), fac: {e: (gy, x)}, st: the state (device tensors, changed in
        place), returns the number of moves"""
        dev = mod.wpacked.device
        N, K, ra = mod.N, mod.K, st["ra"]
        lr, tau, b2, sp, k_ = c["lr_ratio"], c["tau"], c["b2"], c["spend"], int(c["factored"])
        lk = 1 - 1 / tau   # (the fixed leak, core ac6b295)
        th_all = c["theta"]
        es = sorted(fac)
        nf = 0
        if self.seeds == "layer":
            g = torch.Generator(device=dev).manual_seed(13_000_000 + 2 * idx0 + 1_000_003 * step_seed)
            Sr_l = torch.randn(K, k_, device=dev, generator=g) / k_ ** 0.5
            Sc_l = torch.randn(k_, N, device=dev, generator=g) / k_ ** 0.5
        for i0 in range(0, len(es), self.chunk):
            ch = es[i0:i0 + self.chunk]
            for e in ch:      # (per expert: the shapes differ by its token count; the batched form is a speed step)
                gy, x = fac[e]
                st["t"][e] += 1
                t = int(st["t"][e])
                if self.seeds == "core":
                    gen = torch.Generator(device=dev).manual_seed(11_000_000 + 2 * (idx0 + e) + 1_000_003 * (t - 1))
                    Sr = torch.randn(K, k_, device=dev, generator=gen) / k_ ** 0.5
                    Sc = torch.randn(k_, N, device=dev, generator=gen) / k_ ** 0.5
                else:
                    Sr, Sc = Sr_l, Sc_l
                gS = _mm(gy.T, _mm(x, Sr))                                    # g Sr [N, k]
                Sg = _mm(x.T, _mm(gy, Sc.T)).T.contiguous()                   # Sc g [k, K]
                rsq, csq = gS.square().sum(1), Sg.square().sum(0)
                R, C = st["R"][e].to(dev), st["C"][e].to(dev)
                if t > 1:
                    R = b2 * R + (1 - b2) * rsq / K; C = b2 * C + (1 - b2) * csq / N
                else:
                    R, C = (1 - b2) * rsq / K, (1 - b2) * csq / N
                st["R"][e].copy_(R); st["C"][e].copy_(C)
                a = (R / R.mean().clamp_min(1e-30) / (1 - b2 ** t)).sqrt(); b = C.sqrt()
                ia = 1.0 / torch.maximum(a, a.max() * 1e-12 + 1e-30)
                ib = 1.0 / torch.maximum(b, b.max() * 1e-12 + 1e-30)
                th = float(th_all[e]) if torch.is_tensor(th_all) else float(th_all)
                if t == 1:
                    if self.seeds == "core":
                        g0 = torch.Generator(device=dev).manual_seed(4_000_001 + 2 * (idx0 + e))
                    else:
                        g0 = torch.Generator(device=dev).manual_seed(4_000_001 + 2 * (idx0 + e) + 7)
                    V = _orth(-lr * _fg_l(gy, x, ib, ia, torch.randn(N, ra, device=dev, generator=g0)))
                    U = -lr * _fg_r(gy, x, ib, ia, V)
                else:
                    U, V = st["U"][e].to(dev).float(), st["V"][e].to(dev).float()
                    U = U * lk
                    Vn = _orth(V @ (U.T @ U) - lr * _fg_l(gy, x, ib, ia, U))
                    U = U @ (V.T @ Vn) - lr * _fg_r(gy, x, ib, ia, Vn); V = Vn
                A = U @ V.T
                w0 = unpack_rows(mod.wpacked[e], K).to(torch.int8)
                d = A.sign()
                fire = (A.abs() >= th) & ((w0.float() + d).abs() <= 1)
                D = torch.where(fire, d, torch.zeros_like(d))
                nf += int(fire.sum())
                w1 = w0 + D.to(torch.int8)
                mod.wpacked[e].copy_(pack_rows(w1))
                mod._nnz = None
                if mod.beta_nnz:
                    mod.beta[e] = _beta_nnz(w1)
                U = U - sp * (th * (D @ V))
                st["U"][e].copy_(U); st["V"][e].copy_(V)
        mod.invalidate()
        return nf


    @torch.no_grad()
    def update_b(self, mod, fac, st, c, idx0=0, step_seed=0, chunk=None, ret_t=False):
        """update() batched over chunks of experts (seeds "layer" only: one sketch pair per layer and step), the tokens
        padded with zero rows to the chunk's largest count (zero rows add nothing to gy^T x). The same arithmetic as
        update() with seeds "layer", in [C, ...] batches: bmm / batched QR instead of one GEMM per expert. st on the
        device (the caller moves a layer's state in and out whole)."""
        assert self.seeds == "layer"
        with _tf32(False):          # (as update._orth: the Gram products and QR in full fp32)
            _DEFER[0] = True
            try:
                nf = self._update_b(mod, fac, st, c, idx0, step_seed, chunk)
            finally:
                _DEFER[0] = False
            return nf if ret_t or not torch.is_tensor(nf) else int(nf)

    def _repair(self, mod, bad_e, o_h, cnt_h, gy_all, x_all, st, c, idx0, step_seed, chunk, host_t):
        """an expert whose Cholesky broke down has a non-finite U / V; it never fired (a NaN fails |A| >= theta), so
        its trits and nonzero count are untouched. It restarts from this step's gradient as on its first step (state
        zeroed, t 0 -> 1), with Householder QR (6 Nov: rank-1 Grams of 1-2 token experts). Returns its moves."""
        dev = mod.wpacked.device
        ex = bad_e.nonzero().flatten(); ex_h = ex.cpu()
        if not ex_h.numel():
            return 0
        for k in ("U", "V", "R", "C"):
            st[k][ex] = 0
        if host_t:
            st["t"][ex_h] = 0
        else:
            st["t"][ex.to(st["t"].device)] = 0
        ob = torch.zeros(mod.E + 1, dtype=torch.long)
        ob[ex_h + 1] = cnt_h[ex_h].long()
        ob = ob.cumsum(0)
        rows = torch.cat([torch.arange(int(o_h[e]), int(o_h[e + 1])) for e in ex_h.tolist()]).to(dev)
        mod._o_hint = None
        _FORCE_HH[0] = True
        try:
            with _tf32(False):
                nf = self._update_b(mod, (ob.to(dev), gy_all[rows], x_all[rows]), st, c, idx0, step_seed, chunk)
        finally:
            _FORCE_HH[0] = False
        RCHOL_FALLBACK[0] += int(ex_h.numel())
        return nf

    def _update_b(self, mod, fac, st, c, idx0, step_seed, chunk):
        dev = mod.wpacked.device
        N, K, ra = mod.N, mod.K, st["ra"]
        lr, tau, b2, sp, k_ = c["lr_ratio"], c["tau"], c["b2"], c["spend"], int(c["factored"])
        lk = 1 - 1 / tau   # (the fixed leak, core ac6b295)
        th_all = c["theta"]
        C_ = chunk or self.chunk
        g = torch.Generator(device=dev).manual_seed(13_000_000 + 2 * idx0 + 1_000_003 * step_seed)
        Sr = torch.randn(K, k_, device=dev, generator=g) / k_ ** 0.5
        Sc = torch.randn(k_, N, device=dev, generator=g) / k_ ** 0.5
        if fac is None:
            return 0
        if isinstance(fac, dict):            # {e: (gy, x)} -> grouped (offsets, gy, x)
            if not fac:
                return 0
            es_ = sorted(fac)
            o = torch.zeros(mod.E + 1, dtype=torch.long)
            for e in es_:
                o[e + 1] = fac[e][0].shape[0]
            o = o.cumsum(0).to(dev)
            fac = (o, torch.cat([fac[e][0] for e in es_]), torch.cat([fac[e][1] for e in es_]))
        o, gy_all, x_all = fac
        # the counts on the host: one sync per module (E = 1: none, the count is the row count); the same order as
        # on the device (a stable sort on either side gives the same permutation)
        hint = getattr(mod, "_o_hint", None); mod._o_hint = None
        if hint is not None and o.is_cuda:     # (the trainer's counts from the forward, copied long before: no sync)
            hint[1].synchronize(); o_h = hint[0]
        else:
            o_h = (torch.tensor([0, gy_all.shape[0]]) if mod.E == 1 else o.cpu()) if o.is_cuda else o
        o = o.to(dev)
        cnt = o[1:] - o[:-1]
        cnt_h = o_h[1:] - o_h[:-1]
        act_h = torch.nonzero(cnt_h).flatten()
        act_h = act_h[torch.argsort(cnt_h[act_h], stable=True)]       # (similar counts in a chunk: little padding)
        cl = cnt_h[act_h].tolist(); al = act_h.tolist()
        act = _h2d(act_h, dev)
        host_t = not st["t"].is_cuda          # (the trainer keeps t on the host: no per-chunk syncs for "first step?")
        if host_t:
            st["t"][act_h] += 1
            t_all = _h2d(st["t"], dev)
        nf = torch.zeros((), dtype=torch.int64, device=dev)
        bad_e = torch.zeros(mod.E, dtype=torch.bool, device=dev)     # (experts whose rchol broke down this step)

        def _bad(sel):
            if _RB_LAST[0] is not None:
                bad_e[sel] |= _RB_LAST[0]
                _RB_LAST[0] = None
        for i0 in range(0, len(al), C_):
            ch = al[i0:i0 + C_]
            nc = len(ch)
            idx = act[i0:i0 + C_]
            Tm = max(cl[i0:i0 + C_])
            j = torch.arange(Tm, device=dev)
            src = o[idx][:, None] + j[None]                                      # [C, Tm] rows of each expert
            m_ = (j[None] < cnt[idx][:, None])
            src = torch.where(m_, src, torch.zeros_like(src))
            gy = gy_all[src] * m_[..., None].to(gy_all.dtype)                   # [C, Tm, N], zero padded
            x = x_all[src] * m_[..., None].to(x_all.dtype)
            gyT, xT = gy.transpose(1, 2), x.transpose(1, 2)
            if host_t:
                t = t_all[idx]; t_h = st["t"][act_h[i0:i0 + C_]]
            else:
                st["t"][idx.to(st["t"].device)] += 1
                t = st["t"][idx.to(st["t"].device)].to(dev); t_h = None
            gS = _bmmb(gyT, _bmmb(x, Sr.expand(nc, K, k_)))                      # [C, N, k]
            Sg = _bmmb(xT, _bmmb(gy, Sc.T.expand(nc, N, k_))).transpose(1, 2)   # [C, k, K]
            rsq, csq = gS.square().sum(2), Sg.square().sum(1)
            R0, C0 = st["R"][idx].to(dev), st["C"][idx].to(dev)
            first = (t == 1)[:, None]
            if DIAG[0]:      # (the step's gradient energy over its running average, per expert: a spike passes ~x 1/sqrt(1-b2))
                cb_ = (1 - b2 ** (t - 1).clamp_min(1).double()).float()        # (C0 bias-corrected)
                sp_ = ((csq.sum(1) / N) / (C0.sum(1) / cb_).clamp_min(1e-30)).masked_fill(t <= 1, 0.0)
                _dg(mod, "gspike", float(sp_.max())); _dg(mod, "gspike_mean", float(sp_.mean()), "mean")
            R = torch.where(first, (1 - b2) * rsq / K, b2 * R0 + (1 - b2) * rsq / K)
            Cc = torch.where(first, (1 - b2) * csq / N, b2 * C0 + (1 - b2) * csq / N)
            st["R"][idx] = R.to(st["R"].dtype); st["C"][idx] = Cc.to(st["C"].dtype)
            bc = (1 - b2 ** t.double()).float()[:, None]
            a = (R / R.mean(1, keepdim=True).clamp_min(1e-30) / bc).sqrt(); b = Cc.sqrt()
            ia = 1.0 / torch.maximum(a, a.amax(1, keepdim=True) * 1e-12 + 1e-30)
            ib = 1.0 / torch.maximum(b, b.amax(1, keepdim=True) * 1e-12 + 1e-30)
            th = (th_all[idx].to(dev).float() if torch.is_tensor(th_all) else torch.full((nc,), float(th_all), device=dev))
            U = st["U"][idx].to(dev).float(); V = st["V"][idx].to(dev).float()
            ini = ([j for j, v in enumerate(t_h.tolist()) if v == 1] if t_h is not None
                   else (t == 1).nonzero().flatten().tolist())
            if ini:      # experts on their first step: the core's init (a random start basis per expert)
                W0 = torch.stack([torch.randn(N, ra, device=dev, generator=torch.Generator(device=dev).manual_seed(
                    4_000_001 + 2 * (idx0 + ch[j]) + 7)) for j in ini])
                ii = torch.tensor(ini, device=dev)
                Vi = _qr_b(-lr * _bfg_l(gy[ii], x[ii], ib[ii], ia[ii], W0)); _bad(idx[ii])
                Ui = -lr * _bfg_r(gy[ii], x[ii], ib[ii], ia[ii], Vi)
            if len(ini) == 0:            # (the usual chunk: every expert past its first step, no gathers / scatters)
                Ur = U * lk
                with _tf32(TF32_TURN[0]):
                    Pv = torch.baddbmm(-lr * _bfg_l(gy, x, ib, ia, Ur), V, Ur.transpose(1, 2) @ Ur)
                Vn = _qr_b(Pv); del Pv; _bad(idx)
                Fr = _bfg_r(gy, x, ib, ia, Vn)
                if DIAG[0]:  # (the step's update over the accumulator; the basis orthogonality error)
                    rel_ = (lr * Fr.flatten(1).norm(dim=1)) / Ur.flatten(1).norm(dim=1).clamp_min(1e-30)
                    _dg(mod, "upd_rel", float(rel_.max())); _dg(mod, "upd_rel_mean", float(rel_.mean()), "mean")
                    _dg(mod, "upd_max", float((lr * Fr).abs().amax()))
                    _dg(mod, "orth", float((Vn.transpose(1, 2) @ Vn - torch.eye(Vn.shape[2], device=dev)).abs().amax()))
                with _tf32(TF32_TURN[0]):
                    U = torch.baddbmm(-lr * Fr, Ur, V.transpose(1, 2) @ Vn)
                V = Vn
            elif (rest := (torch.tensor([j for j, v in enumerate(t_h.tolist()) if v > 1], dtype=torch.long, device=dev)
                           if t_h is not None else (t > 1).nonzero().flatten())).numel():   # (host t: no sync)
                Ur, Vr = U[rest] * lk, V[rest]
                with _tf32(TF32_TURN[0]):
                    Pv = Vr @ (Ur.transpose(1, 2) @ Ur) - lr * _bfg_l(gy[rest], x[rest], ib[rest], ia[rest], Ur)
                Vn = _qr_b(Pv); del Pv; _bad(idx[rest])
                with _tf32(TF32_TURN[0]):
                    U[rest] = Ur @ (Vr.transpose(1, 2) @ Vn) - lr * _bfg_r(gy[rest], x[rest], ib[rest], ia[rest], Vn)
                V[rest] = Vn
            if ini:
                U[ii] = Ui; V[ii] = Vi
            if FIRE_B[0] == "kernel" and dev.type == "cuda" and not torch.is_tensor(th_all):
                # the core's ttf_fire per expert: A from bf16 copies with an error bound, the uncertain entries in fp32,
                # the moves applied to the packed trits in place, D V for the spend (no [C, N, K] fp32 A, no unpack)
                thf = float(th_all)
                if mod.beta_nnz and mod._nnz is None:    # (the nonzeros per expert, counted once, then kept by the moves)
                    mod._nnz = torch.cat([unpack3(mod.wpacked[e0:e0 + 64], K).ne(0).sum((1, 2))
                                          for e0 in range(0, mod.E, 64)])
                if _NANDUMP and not _NANDUMP_DONE:
                    _nan_dump(mod, st, idx, t, gy, x, ia, ib, R, Cc, R0, C0, U, V, lr, lk, th, ini)
                dU, cnt_ = GGK.bfire(U, V, mod.wpacked, idx, K, thf)     # (batched over the chunk's experts)
                U = U - sp * thf * dU
                nf += cnt_[:, 0].sum()
                if mod.beta_nnz:
                    mod._nnz[idx] += cnt_[:, 1].to(mod._nnz.dtype)
                    mod.beta[idx] = (mod._nnz[idx].float() / N).clamp_min(1.0).rsqrt()
                st["U"][idx] = U.to(st["U"].dtype); st["V"][idx] = V.to(st["V"].dtype)
                continue
            A = U @ V.transpose(1, 2)                                            # [C, N, K]
            fire = A.abs() >= th[:, None, None]
            d = A.sign().to(torch.int8)
            del A                       # (one fp32 [C, N, K] at a time; the rest int8 / bool)
            w0 = unpack3(mod.wpacked[idx], K)
            fire &= (w0 + d).abs() <= 1
            D = d * fire
            del d
            nf += torch.count_nonzero(fire)          # (no int64 copy of the mask)
            del fire
            w1 = w0 + D
            del w0
            mod.wpacked[idx] = pack3(w1)
            mod._nnz = None
            if mod.beta_nnz:
                mod.beta[idx] = (w1.ne(0).sum((1, 2)).float() / N).clamp_min(1.0).rsqrt()
            del w1
            U = U - sp * th[:, None, None] * (D.float() @ V)
            del D
            st["U"][idx] = U.to(st["U"].dtype); st["V"][idx] = V.to(st["V"].dtype)
        if not _FORCE_HH[0]:
            p_ = (mod, bad_e, o_h, cnt_h, gy_all, x_all, st, c, idx0, step_seed, chunk, host_t)
            if DEFER_LAYER[0]:      # (the trainer checks a whole layer at once: finish_repairs)
                _PEND.append((self,) + p_)
            elif bool(bad_e.any()):     # (one host check per module)
                nf = nf + self._repair(*p_)
        mod.invalidate()
        return nf

def _mm(A, B):
    """update._mmb (2-D): bf16 A, B rounded to bf16, fp32 accumulation"""
    if A.dtype != torch.bfloat16:
        return A @ B
    if not A.is_cuda:
        return A.float() @ B.to(A.dtype).float()
    return torch.mm(A, B.to(A.dtype), out_dtype=torch.float32)


def _fg_r(gy, x, b, a, W):
    """u W / (-lr) for u = -lr diag(a) gy^T x diag(b): a * (gy^T (x (b W)))  (update._fg_mm_r)"""
    return a[:, None] * _mm(gy.T, _mm(x, b[:, None] * W))


def _fg_l(gy, x, b, a, W):
    """u^T W / (-lr): b * (x^T (gy (a W)))  (update._fg_mm_l)"""
    return b[:, None] * _mm(x.T, _mm(gy, a[:, None] * W))


def _bfg_r(gy, x, b, a, W):
    """_fg_r batched: a * (gy^T (x (b W))), gy [C, T, N], x [C, T, K], b [C, K], a [C, N], W [C, K, r]"""
    return a[:, :, None] * _bmmb(gy.transpose(1, 2), _bmmb(x, b[:, :, None] * W))


def _bfg_l(gy, x, b, a, W):
    """_fg_l batched: b * (x^T (gy (a W))), W [C, N, r]"""
    return b[:, :, None] * _bmmb(x.transpose(1, 2), _bmmb(gy, a[:, :, None] * W))
