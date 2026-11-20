"""The ternary layer of TTF: KernelTernaryLinear, pure trits packed 5 per byte, the forward on the packed-ternary GEMM.

Its weight gradient is never applied here: with capture on (every TTF run) the backward hands it, or with --dw factored
its factors (gy, x), to the layer's on_grad hook (the TTF update, update.ts_layer), or keeps it in .gw when the layer
has no hook (analysis tools). The earlier rules that flipped trits inside this backward (the stateless stochastic flip,
the evidence counter, lockout, error feedback, the flip predictor) are at tag pre-consolidation-2026-10-17.
"""
from __future__ import annotations
import contextlib
import math
import torch
import torch.nn as nn

from .config import ModelConfig
from .model import BitTransformer

_PROF_ON = torch._C._autograd._profiler_enabled
_NULL = contextlib.nullcontext()


def _rf(name):
    """a named range for TERN_PROF. torch.profiler.record_function costs two dispatcher calls even with no profiler
    running (~60k per 0.8B adapter step), so outside a profile it is a shared no-op context"""
    return torch.profiler.record_function(name) if _PROF_ON() else _NULL


from .kernel import (pack_rows, unpack_fast, tern_gemm_dx, trit_nnz, beta_from_nnz,
                     trit_diff, popcount, tern_gemm_i8, tern_gemm_dx_i8x2, act_quant_i8, act_quant_i8_fast)
from . import kernel as _K


class _ColScaleFn(torch.autograd.Function):
    """x * c (c per input column, cast to x's dtype) with the backward in one kernel: gx = g * c and the column sums of
    g * x (each product rounded to x's dtype as torch's mul backward does; the sum in fp32, its order differs)"""
    @staticmethod
    def forward(ctx, x, c):
        cb = c.to(x.dtype)
        ctx.save_for_backward(x, cb); ctx.cdt = c.dtype
        return x * cb

    @staticmethod
    def backward(ctx, g):
        from .kernel import scale_bwd
        x, cb = ctx.saved_tensors
        K = x.shape[-1]
        gx, gc = scale_bwd(g.reshape(-1, K).contiguous(), cb, x.reshape(-1, K).contiguous())
        return gx.view(x.shape), gc.to(cb.dtype).to(ctx.cdt)


class _AddRowFn(torch.autograd.Function):
    """(y + a) * r (r per output row) in one kernel, the backward in one more: gy = ga = g * r and the column sums of
    g * (y + a) for r (the same per-element roundings as the unfused add and mul)"""
    @staticmethod
    def forward(ctx, y, a, r):
        from .kernel import addrow_fwd
        rb = r.to(y.dtype)
        N = y.shape[-1]
        t, o = addrow_fwd(y.reshape(-1, N).contiguous(), a.reshape(-1, N).contiguous(), rb)
        ctx.save_for_backward(t, rb); ctx.rdt = r.dtype; ctx.shape = y.shape
        return o.view(y.shape)

    @staticmethod
    def backward(ctx, g):
        from .kernel import scale_bwd
        t, rb = ctx.saved_tensors
        gt, gr = scale_bwd(g.reshape(-1, t.shape[-1]).contiguous(), rb, t)
        gt = gt.view(ctx.shape)
        return gt, gt, gr.to(rb.dtype).to(ctx.rdt)


FACTORED = [False]           # --dw factored: on for this step (off on --ttf-vdiag steps, which read dense g)
DX_MODE = ["bf16"]             # --dx int8x2: the input gradient on the int8 tensor cores, gy in two int8 slices (int8: one)
GRAD_BITS = [0]                # --grad-bits b: gy (the gradient entering each ternary product) in b-bit integers, below
GRAD_GROUP = [0]               # --grad-group G: one scale per G consecutive entries of a row instead of one per row
GRAD_GRID = ["zero"]           # --grad-grid odd: all 2^b codes as odd integers, no zero level (see grad_quant)


def grad_quant(gy, bits, group=0):
    """--grad-bits (18 Nov 2026, user: "a from scratch 110M ternary with 4 bit and 2 bit gradients"): gy [M, N] -> (qi int8
    [M, N], s fp32 [M] or [M, N / G], gq bf16 [M, N] = bf16(qi * s)). Absmax scale per row (token), or per G consecutive
    entries (--grad-group G), stochastic rounding: unbiased, E[gq] = gy. Both backward products use the quantized gy: the
    input gradient (qi against the trits) and the TTF update's factor (gq).
    Grid (GRAD_GRID): "zero" = the integers -qmax..qmax, qmax = 2^(b-1) - 1 (4 bit: -7..7; 2 bit: -1, 0, 1 = 3 of the 4
    codes). "odd" (18 Nov, user: "1 bit is sign, and the other bit could be literally 1 and 2"): all 2^b codes, the odd
    integers -(2^b - 1)..(2^b - 1) (2 bit: -3, -1, 1, 3; 4 bit: -15..15), no zero; step 2 / (2^b - 1) of the scale against
    1 / qmax: 2-bit rounding variance down to (2/3)^2 = 0.44x."""
    odd = GRAD_GRID[0] == "odd"
    qmax = (1 << bits) - 1 if odd else (1 << (bits - 1)) - 1
    g = gy.float()
    if group:   # --grad-group G (18 Nov: per-token 2-bit was ~4x less step-efficient; one outlier set a row's scale)
        M, N = gy.shape   # the caller forms the input gradient from gq in bf16 (no per-row int8 path)
        if N % group:
            raise ValueError(f"--grad-group {group}: a ternary layer's output width {N} is not a multiple of it")
        g = g.view(M, N // group, group)
    s = g.abs().amax(-1).clamp_min(1e-30) / qmax
    t = g / s[..., None]
    if odd:   # q = 2k - 1, k = floor((t + 1) / 2 + U): E[q] = t
        qi = (2 * torch.floor((t + 1) * 0.5 + torch.rand_like(t)) - 1).clamp_(-qmax, qmax)
    else:
        qi = torch.floor(t + torch.rand_like(t)).clamp_(-qmax, qmax)
    gq = (qi * s[..., None]).view(gy.shape).to(torch.bfloat16)
    return qi.view(gy.shape).to(torch.int8), s, gq


class _KernelTernFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wpacked, layer):
        beta = layer._cached_beta(wpacked) if layer.use_beta else None
        # activations are already 8-bit, so x @ W^T is exact in int32 on the s8 tensor cores, and the saved
        # activation is 1 byte per element
        xq, xs = (act_quant_i8_fast if x.is_cuda else act_quant_i8)(x.reshape(-1, layer.K), layer.act_bits)
        with _rf("tern.fwd"):
            y = tern_gemm_i8(xq, xs, wpacked, layer.K, beta if beta is not None else 1.0,
                             layer.w8() if (W8_CACHE[0] and wpacked.data_ptr() == layer.wpacked.data_ptr()) else None)
        y = y.view(*x.shape[:-1], layer.N)
        ctx.save_for_backward(xq, xs, wpacked)
        ctx.layer = layer
        ctx.beta = beta
        ctx.xshape = x.shape
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        layer = ctx.layer
        beta = ctx.beta
        gyf = gy.reshape(-1, layer.N).contiguous()
        if beta is not None:
            gyf = gyf * beta.to(gyf.dtype)       # y = beta * x@W^T: both grads carry beta
        xq, xs, wpacked = ctx.saved_tensors
        if GRAD_BITS[0]:
            qi, qs, gyf = grad_quant(gyf, GRAD_BITS[0], GRAD_GROUP[0])
        with _rf("tern.dx"):
            if GRAD_BITS[0] and GRAD_GROUP[0]:
                gx = tern_gemm_dx(gyf, wpacked, layer.K).view(ctx.xshape)
            elif GRAD_BITS[0] and gyf.is_cuda:   # qi against the trits, exact: bf16((qi W) s)
                w8t = layer.w8(t=True) if wpacked.data_ptr() == layer.wpacked.data_ptr() else _K.unpack_t_i8(wpacked, layer.K)
                gx = (torch.mm(qi.float(), w8t.t().float()) * qs[:, None]).to(torch.bfloat16).view(ctx.xshape)
            else:
                gx = _grad_x(layer, gyf, wpacked).view(ctx.xshape)
        if not layer.capture:     # no update asked for: the input gradient only (the trits stay as they are)
            return gx.to(gy.dtype), None, None
        if getattr(layer, "cap_accum", False):   # --accum > 1: the update sees every micro-batch (no gradient kept
            layer._ts_final = getattr(layer, "cap_final", True)   # across them); the firing on the last one
            layer._ts_micro = getattr(layer, "cap_micro", None)   # (the index: update._micro_t)
        xf = (_K.dequant_i8(xq, xs) if xq.is_cuda and gyf.dtype == torch.bfloat16 and xq.dim() == 2
              else xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype))   # (one kernel, the same bf16 roundings)
        if FACTORED[0] and getattr(layer, "on_grad", None) is not None and not getattr(layer, "_dw_dense", False):
            layer.on_grad(layer, (gyf, xf))      # --dw factored: no dense weight gradient, the factors (gy, x), g = gy^T x
            return gx.to(gy.dtype), None, None
        with _rf("tern.dw"):
            if getattr(layer, "_dw_dense", False):   # --dw-thin dense: a thin layer's dense gradient (2 T N K flops <
                gw = torch.mm(gyf.transpose(0, 1), xf, out_dtype=torch.float32)   # the factored 2 T (N + K)(k + 2 r))
            else:
                gw = (gyf.transpose(0, 1) @ xf).float()
        if getattr(layer, "on_grad", None) is not None:
            # the layer's update runs here, after its input gradient (gx) is computed, so this layer's trits are not
            # needed again in this step (earlier layers recompute with their own, unchanged trits)
            layer.on_grad(layer, gw)
            return gx.to(gy.dtype), None, None
        layer.gw = gw              # no update hook: kept for the caller (analysis tools)
        if getattr(layer, "capture_h", False):
            # Gauss-Newton diagonal for this layer: H[n,k] = sum_m (beta*gy)^2 x^2.
            # Same shape and cost as the weight gradient, one extra GEMM.
            layer.hw = ((gyf * gyf).transpose(0, 1) @ (xf * xf)).detach().float()
        return gx.to(gy.dtype), None, None


def _grad_x(layer, gyf, wpacked):
    if DX_MODE[0] in ("int8x2", "int8"):
        return tern_gemm_dx_i8x2(gyf, wpacked, layer.K, two=DX_MODE[0] == "int8x2",
                                 w8t=layer.w8(t=True) if (W8_CACHE[0] and wpacked.data_ptr() == layer.wpacked.data_ptr()) else None)
    return tern_gemm_dx(gyf, wpacked, layer.K)


W8_CACHE = [False]       # KernelTernaryLinear.w8: the int8 trits kept between their changes (turned on by --ckpt-skip
                         # auto when the memory allows: 2 bytes per trit; False: unpacked per use)
GRAPH_INPLACE = [False]  # --cuda-graph: after a firing the cached beta and int8 trits are rewritten in place (_graph_refresh)
FUSED_LAYER = [False]    # --fused-layer: centring, column scale, act-quant, GEMM epilogue, row scale, bias and the adapter
                         # sum in a few kernels each way (_FusedTernFn); bit-identical output and input grad


def _fused_ok(l, x):
    """the layers _FusedTernFn takes: bf16 CUDA input, row / column scales, at most the additive float adapter; with
    capture on, only the factored TTF update (--dw factored) and no --dw-thin dense"""
    if not (FUSED_LAYER[0] and x.is_cuda and x.dtype == torch.bfloat16
            and l.col_scale is not None and l.row_scale is not None
            and (getattr(l, "mag_A", None) is None or getattr(l, "mag_kind", "add") == "add")
            and l.K % 16 == 0 and l.N % 8 == 0 and x.numel() // l.K > 16):   # (K % 16: kernel.tern_gemm_i8)
        return False
    if l.capture and torch.is_grad_enabled():
        return FACTORED[0] and getattr(l, "on_grad", None) is not None and not getattr(l, "_dw_dense", False)
    return True


def _fused_fwd(l, x2, acc2, save):
    """x2 [M, K] -> (out [M, N] bf16, saved tensors or None). With the float adapter (mag_A, mag_B): its term
    a = (xs B) A^T on the scaled input xs = bf16(bf16(x - mu) c) (the prologue writes xs), added to the ternary product
    before the row scale in the GEMM's epilogue: the unfused path's (_ColScaleFn, two GEMMs, _AddRowFn) values"""
    mu = getattr(l, "in_mean", None)
    mu_b = mu.to(torch.bfloat16) if mu is not None else None
    cb, rb = l.col_scale.to(torch.bfloat16), l.row_scale.to(torch.bfloat16)
    ob = getattr(l, "out_bias", None)
    bb = ob.to(torch.bfloat16) if ob is not None else None
    # (the input itself is kept for the backward, which re-forms bf16(x - mu): one tensor for all the readers of one
    # input, bit-identical)
    mag = getattr(l, "mag_A", None) is not None
    if mag:
        xq, xs, _, xo = _K.fused_prologue(x2, mu_b, cb, l.act_bits, False, True)
        Ab, Bb = l.mag_A.to(torch.bfloat16), l.mag_B.to(torch.bfloat16)
        h = xo @ Bb
        a = h @ Ab.T
    else:
        xq, xs, _ = _K.fused_prologue(x2, mu_b, cb, l.act_bits, False)
        a = None
    beta = l._cached_beta(l.wpacked) if l.use_beta else None
    t, o = _K.fused_fwd_gemm(xq, l.w8(), xs, beta if beta is not None else 1.0, rb, bb, acc2, save, a)
    if not save:
        return o, None
    return o, (xq, xs, x2, t, cb, rb, beta) + ((xo, h, Ab, Bb) if mag else ())


class _FusedTernFn(torch.autograd.Function):
    """out = acc + ((T(c (x - mu)) * beta) * r + bias), the ternary layer with its scales, centring and an added input
    (the sum of a converted layer's parts) as one Function; the TTF update gets the same (gy, x) factors"""
    @staticmethod
    def forward(ctx, x, c, r, bias, acc, layer, mA=None, mB=None):
        x2 = x.reshape(-1, layer.K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        acc2 = acc.reshape(-1, layer.N).contiguous() if acc is not None else None
        with _rf("tern.fwd"):
            o, sv = _fused_fwd(layer, x2, acc2, True)
        xq, xs, xc, t, cb, rb, beta = sv[:7]
        ctx.has_mag = mA is not None
        if ctx.has_mag:
            ctx.mdt = (mA.dtype, mB.dtype)
        ctx.mu_b = (layer.in_mean.to(torch.bfloat16) if getattr(layer, "in_mean", None) is not None
                    else None)   # (xc above is the input, its centring redone in the backward)
        ctx.save_for_backward(xq, xs, xc, t, cb, rb, *sv[7:])
        ctx.layer, ctx.beta, ctx.xshape = layer, beta, x.shape
        ctx.cdt, ctx.rdt, ctx.has_bias, ctx.has_acc = c.dtype, r.dtype, bias is not None, acc is not None
        ctx.bdt = bias.dtype if bias is not None else None
        return o.view(*x.shape[:-1], layer.N)

    @staticmethod
    def backward(ctx, g):
        sv = ctx.saved_tensors          # (read once: a checkpointed layer's saved tensors unpack only once)
        xq, xs, xc, t, cb, rb = sv[:6]
        l = ctx.layer
        g2 = g.reshape(-1, l.N).contiguous()
        bb = (ctx.beta if ctx.beta is not None else torch.ones((), device=g.device)).reshape(()).to(torch.bfloat16)
        sl = {"int8x2": 2, "int8": 1}.get(DX_MODE[0], 0)
        gA = gB = gxa = None
        with _rf("tern.dx"):
            lr = None
            if ctx.has_mag and sl:   # the float adapter a = (xs B) A^T: gh = bf16(g r) A from the row pass, its input
                xo, h, Ab, Bb = sv[6:]   # gradient gh B^T formed inside the dx kernel (no [M, K] round trip)
                f, q1, q2, s, gbias, grow, gt, gh = _K.fused_bwd_rows(g2, t, rb, bb, sl, save_gt=True, a_b=Ab)
                gA = torch.mm(h.t(), gt).t().to(ctx.mdt[0])
                gB = torch.mm(xo.t(), gh).to(ctx.mdt[1])
                lr = (gh, Bb)
                del gt
            else:
                f, q1, q2, s, gbias, grow, *gt = _K.fused_bwd_rows(g2, t, rb, bb, sl, save_gt=ctx.has_mag)
            if GRAD_BITS[0]:   # --grad-bits: f (the gradient entering the ternary product) quantized; the float adapter's
                q1, s, f = grad_quant(f, GRAD_BITS[0], GRAD_GROUP[0])   # own gradients (gt, gh) stay bf16
                q2 = None
                if not sl or GRAD_GROUP[0]:   # (bf16 dx / group scales: the quantized f through the bf16 dx GEMM)
                    q1 = s = None
            if ctx.has_mag and lr is None:   # (one slice off: the adapter's gradients as autograd's mm backward forms them)
                xo, h, Ab, Bb = sv[6:]
                gt = gt[0]                                    # bf16(g * r)
                gA = torch.mm(h.t(), gt).t().to(ctx.mdt[0])
                gh = torch.mm(gt, Ab)
                gB = torch.mm(xo.t(), gh).to(ctx.mdt[1])
                gxa = torch.mm(gh, Bb.t())                    # (added to the ternary input gradient before the scale)
                del gt, gh
            if sl and q1 is not None:
                gx, gc = _K.dx2_col(q1, q2, s, l.w8(t=True), cb, xc, ctx.mu_b, None, lr)
                del q1, q2
            else:
                gxt = tern_gemm_dx(f, l.wpacked, l.K) if (GRAD_BITS[0] and GRAD_GROUP[0]) else _grad_x(l, f, l.wpacked)
                if lr is not None:   # (the float adapter's input gradient, formed in dx2_col on the int8 path)
                    gxt = gxt + torch.mm(lr[0], lr[1].t())
                gx, gc = _K.scale_bwd(gxt if gxa is None else gxt + gxa, cb, xc, ctx.mu_b)   # (the centring in the kernel)
        if l.capture and getattr(l, "on_grad", None) is not None:    # (_fused_ok: the factored TTF update)
            if getattr(l, "cap_accum", False):
                l._ts_final = getattr(l, "cap_final", True); l._ts_micro = getattr(l, "cap_micro", None)
            l.on_grad(l, (f, _K.dequant_i8(xq, xs)))
        return (gx.view(ctx.xshape), gc.to(ctx.cdt), grow.to(ctx.rdt), gbias.to(ctx.bdt) if ctx.has_bias else None,
                g if ctx.has_acc else None, None, gA, gB)


class KernelTernaryLinear(nn.Module):
    """Pure-ternary linear, forward on the Triton packed-ternary GEMM (weights
    stay packed, never materialized dense). Row-packed 5-trits/byte along K."""
    def __init__(self, in_features, out_features, act_bits: int = 8, beta: bool = False):
        super().__init__()
        self.use_beta = beta      # scale output by 1/sqrt(K*rho); False = legacy raw trits
        self.K = in_features
        self.N = out_features
        self.act_bits = act_bits
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        scale = w.abs().mean().clamp_min(1e-5)
        weight = (w / scale).round().clamp_(-1, 1).to(torch.int8)
        self.register_buffer("wpacked", pack_rows(weight))       # [N, ceil(K/5)]
        # --- optional learned per-row / per-column scales (enable_rc_scale) -------
        # effective weight W_ij = beta * r_i * c_j * q_ij, from N + K floats per layer
        self.row_scale = None
        self.col_scale = None
        # --- the weight gradient goes to the update (on_grad) or to .gw instead of being dropped ---
        self.capture = False
        self.capture_h = False
        self.gw = None
        self.hw = None
        # unused, kept so that checkpoints written before 17 Oct 2026 (which have it) load unchanged
        self.register_buffer("frozen_scale", torch.zeros(()), persistent=True)
        # --- flip instrumentation (off unless enable_tracking() is called) ------
        self.track = False
        self.touched = None      # uint8 [N,K5] sticky mask: trit ever changed
        self.flips = None        # 0-dim int64: trit changes since last reset

    def _wkey(self, wpacked):
        return (wpacked.data_ptr(), wpacked._version)

    @torch.no_grad()
    def _cached_beta(self, wpacked):
        # beta is computed once per change of the trits (an update's copy_ bumps wpacked's version), not per forward
        k = self._wkey(wpacked)
        if getattr(self, "_beta_key", None) != k:
            self._nnz = trit_nnz(wpacked, self.K); self._beta_val = beta_from_nnz(self._nnz, self.N); self._beta_key = k
        return self._beta_val

    def w8(self, t=False):
        """the trits as int8 for the step's GEMMs, [N, K] (t: the [K, N] copy dx reads), unpacked once per change of the
        trits and kept until the next (the fused firing writes them in place: _after_fire counts it) instead of on every
        forward, recompute and dx (27B FFN: 16 micro-batches x 3 unpacks a step). W8_CACHE False: unpacked per use"""
        if not W8_CACHE[0]:
            return _K.unpack_t_i8(self.wpacked, self.K) if t else unpack_fast(self.wpacked, self.K)
        k = (self._wkey(self.wpacked), getattr(self, "_nfire", 0))
        c = self.__dict__.setdefault("_w8c", {})
        if t not in c or c[t][0] != k:
            c.pop(t, None)
            c[t] = (k, _K.unpack_t_i8(self.wpacked, self.K) if t else unpack_fast(self.wpacked, self.K))
        return c[t][1]

    def _after_fire(self, dnnz):
        """after the fused firing (trits written in place, no version bump): the nonzero count moves by dnnz, so a
        valid cached beta is updated exactly instead of recomputed from all the trits"""
        self._nfire = getattr(self, "_nfire", 0) + 1
        if GRAPH_INPLACE[0]:
            return self._graph_refresh()
        if getattr(self, "_beta_key", None) is not None and self._beta_key == self._wkey(self.wpacked):
            self._nnz = self._nnz + dnnz; self._beta_val = beta_from_nnz(self._nnz, self.N)
        else:
            self._beta_key = None

    @torch.no_grad()
    def _graph_refresh(self):
        """--cuda-graph, after a firing inside a captured step: the cached beta (from all the trits: the same integer
        count as the running one) and the cached int8 trits rewritten in their own tensors, so the step's forward,
        replayed later, reads the moved trits; the cache keys follow the trits' version (dense_fire's copy_ moves it)"""
        if getattr(self, "_beta_key", None) is not None:
            nnz = trit_nnz(self.wpacked, self.K)
            if torch.is_tensor(self._nnz):
                self._nnz.copy_(nnz)
            else:
                self._nnz = nnz
            self._beta_val.copy_(beta_from_nnz(nnz, self.N))
            self._beta_key = self._wkey(self.wpacked)
        c = self.__dict__.get("_w8c")
        if c:
            k = (self._wkey(self.wpacked), getattr(self, "_nfire", 0))
            for t_, (_, buf) in list(c.items()):
                buf.copy_(_K.unpack_t_i8(self.wpacked, self.K) if t_ else unpack_fast(self.wpacked, self.K))
                c[t_] = (k, buf)

    def _invalidate(self):
        """drop the cached beta and int8 weight copy; call after writing wpacked from outside the update (ttfconv's
        model / finish / swap / blocklocal, ttf-home's ref_ppl)"""
        self._beta_key = None
        self.__dict__.pop("_w8c", None)

    def enable_tracking(self):
        """Track per-step flip count and which trits have ever changed since init.

        Costs one packed-size uint8 mask (~1.6 bit/weight) plus a packed clone per update.
        """
        self.track = True
        dev = self.wpacked.device
        self.touched = torch.zeros_like(self.wpacked)
        self.flips = torch.zeros((), dtype=torch.int64, device=dev)

    @torch.no_grad()
    def _record_flips(self, before, after):
        d = trit_diff(before, after)
        self.touched |= d
        self.flips += popcount(d)

    @torch.no_grad()
    def flip_stats(self):
        """(flips since reset, never-changed trits) as 0-dim int64 GPU tensors."""
        pad = self.N * (self.wpacked.shape[1] * 5 - self.K)
        never = (self.N * self.K + pad) - popcount(self.touched) - pad
        return self.flips, never

    def enable_rc_scale(self):
        dev = self.wpacked.device
        self.row_scale = nn.Parameter(torch.ones(self.N, device=dev))
        self.col_scale = nn.Parameter(torch.ones(self.K, device=dev))

    def enable_lowrank_mag(self, kind, r):
        """Low-rank float adapter beyond the trits: W = beta*T + A B^T, r (N + K) floats; A starts at 0 (the layer is
        unchanged at first; a converted checkpoint's adapters load with its weights)"""
        dev = self.wpacked.device
        self.mag_kind = kind
        assert kind == "add", "the adapter is additive (the multiplicative and ternary kinds: tag pre-consolidation-2026-10-17)"
        self.mag_A = nn.Parameter(torch.zeros(self.N, r, device=dev))
        std = (1.0 / self.K) ** 0.5
        self.mag_B = nn.Parameter(torch.randn(self.K, r, device=dev) * std)
        return [self.mag_A, self.mag_B]

    def enable_centering(self, mu, bias):
        """centred layer (a gptqc conversion): y = layer(x - mu) + bias, mu a fixed buffer, bias a float parameter (init W mu)"""
        self.register_buffer("in_mean", mu.detach().clone().float())
        self.out_bias = nn.Parameter(bias.detach().clone().float())

    def forward(self, x, acc=None):
        if _fused_ok(self, x):    # --fused-layer (acc: a tensor added to the output, the sum of a converted layer's parts)
            if not torch.is_grad_enabled():
                x2 = x.reshape(-1, self.K)
                o, _ = _fused_fwd(self, x2 if x2.is_contiguous() else x2.contiguous(),
                                  acc.reshape(-1, self.N).contiguous() if acc is not None else None, False)
                return o.view(*x.shape[:-1], self.N)
            return _FusedTernFn.apply(x, self.col_scale, self.row_scale, getattr(self, "out_bias", None), acc, self,
                                      getattr(self, "mag_A", None), getattr(self, "mag_B", None))
        y = self._forward(x)
        return y if acc is None else acc + y

    def _forward(self, x):
        if getattr(self, "in_mean", None) is not None:      # gptqc: centred input (before the activation quantisation)
            x = x - self.in_mean.to(x.dtype)
        # column scale before the GEMM (and before the int8 activation quantization),
        # row scale after it: the packed kernels are unchanged, autograd handles r and c,
        # and the update's dL/dq comes out scaled by r_i * c_j per weight
        fz = (x.is_cuda and x.dtype == torch.bfloat16 and self.col_scale is not None
              and self.row_scale is not None and getattr(self, "mag_A", None) is not None)
        if fz:      # the scales' and the adapter's elementwise work in fused kernels (the same values)
            x = _ColScaleFn.apply(x, self.col_scale)
            y = _KernelTernFn.apply(x, self.wpacked, self)
            a = (x @ self.mag_B.to(x.dtype)) @ self.mag_A.to(y.dtype).T
            y = _AddRowFn.apply(y, a, self.row_scale)
            if getattr(self, "out_bias", None) is not None:
                y = y + self.out_bias.to(y.dtype)
            return y
        if self.col_scale is not None:
            x = x * self.col_scale.to(x.dtype)
        y = _KernelTernFn.apply(x, self.wpacked, self)
        if getattr(self, "mag_A", None) is not None:
            y = y + (x @ self.mag_B.to(x.dtype)) @ self.mag_A.to(y.dtype).T
        if self.row_scale is not None:
            y = y * self.row_scale.to(y.dtype)
        if getattr(self, "out_bias", None) is not None:
            y = y + self.out_bias.to(y.dtype)
        return y


def build_kernel_transformer(c: ModelConfig, grad_checkpoint: bool = True, beta: bool = False):
    def make_linear(i, o):
        return KernelTernaryLinear(i, o, c.act_bits, beta=beta)
    return BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)


def enable_rc_scales(model):
    """Learned per-row and per-column scales on every kernel ternary layer."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.enable_rc_scale(); n += m.N + m.K
    return n


def enable_flip_tracking(model):
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.enable_tracking(); n += 1
    return n


@torch.no_grad()
def collect_flip_stats(model, reset: bool = True):
    """Per-layer flip stats for one step. One host sync for the whole model."""
    names, flips, never, numels = [], [], [], []
    for name, m in model.named_modules():
        if isinstance(m, KernelTernaryLinear) and m.track:
            f, nv = m.flip_stats()
            names.append(name); flips.append(f); never.append(nv)
            numels.append(m.N * m.K)
    if not names:
        return {}
    f = torch.stack(flips).cpu().tolist()          # single sync
    nv = torch.stack(never).cpu().tolist()
    if reset:
        for m in model.modules():
            if isinstance(m, KernelTernaryLinear) and m.track:
                m.flips.zero_()
    tot = sum(numels)
    return {
        "layers": names,
        "flip_frac": [fi / n for fi, n in zip(f, numels)],
        "never_frac": [ni / n for ni, n in zip(nv, numels)],
        "flip_frac_total": sum(f) / tot,
        "never_frac_total": sum(nv) / tot,
    }
