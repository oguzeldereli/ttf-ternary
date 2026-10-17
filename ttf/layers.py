"""The ternary layer of TTF: KernelTernaryLinear, pure trits packed 5 per byte, the forward on the packed-ternary GEMM.

Its weight gradient is never applied here: with capture on (every TTF run) the backward hands it, or with --dw factored
its factors (gy, x), to the layer's on_grad hook (the fused TTF update, train.ts_layer), or keeps it in .gw for the
update after the backward. The earlier rules that flipped trits inside this backward (the stateless stochastic flip,
the evidence counter, lockout, error feedback, the flip predictor) are in ttf.legacy.flip.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.profiler import record_function as _rf      # named ranges for TERN_PROF (cheap when not profiling)

from .config import ModelConfig
from .model import BitTransformer


from .kernel import (pack_rows, unpack_rows, tern_gemm, tern_gemm_dx, trit_beta, trit_diff, popcount,
                     tern_gemm_i8, tern_gemm_dx_i8, act_quant_i8, act_quant_i8_fast)


def _act_quant_plain(x, bits=8):
    """Detached 8-bit activation quant (no autograd graph); used inside the
    custom Function where the straight-through path is handled manually."""
    qp = 2 ** (bits - 1) - 1
    scale = qp / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (x * scale).round().clamp_(-qp - 1, qp) / scale


FUSED_SCALES = [False]       # --fused-scales: the column scale's backward and the adapter add + row scale in fused kernels


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


class _KernelTernFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wpacked, layer):
        beta = layer._cached_beta(wpacked) if layer.use_beta else None
        if layer.int8:
            # int8 path: activations are already 8-bit, so x @ W^T is exact in int32
            # on the s8 tensor cores, and the saved activation is 1 byte/element.
            xq, xs = (act_quant_i8_fast if x.is_cuda else act_quant_i8)(x.reshape(-1, layer.K), layer.act_bits)
            with _rf("tern.fwd"):
                y = tern_gemm_i8(xq, xs, wpacked, layer.K, beta if beta is not None else 1.0)
            y = y.view(*x.shape[:-1], layer.N)
            ctx.save_for_backward(xq, xs, wpacked)
        else:
            xq = _act_quant_plain(x, layer.act_bits)
            y = tern_gemm(xq.reshape(-1, layer.K), wpacked, layer.K).view(
                *x.shape[:-1], layer.N)
            if beta is not None:
                y = y * beta.to(y.dtype)
            ctx.save_for_backward(x, wpacked)
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
        if (FACTORED[0] and layer.int8 and layer.capture and getattr(layer, "on_grad", None) is not None
                and (getattr(layer, "ts_micro", False) or not getattr(layer, "cap_accum", False))):
            # --dw factored: no dense weight gradient; the TTF update gets its factors (gy, x), g = gy^T x
            xq, xs, wpacked = ctx.saved_tensors
            with _rf("tern.dx"):
                gx = _grad_x(layer, gyf, wpacked).view(ctx.xshape)
            if getattr(layer, "cap_accum", False):
                layer._ts_final = getattr(layer, "cap_final", True)
            layer.on_grad(layer, (gyf, xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype)))
            return gx.to(gy.dtype), None, None
        if not layer.capture:     # no update asked for: the input gradient only (the trits stay as they are)
            wpacked = ctx.saved_tensors[-1]
            with _rf("tern.dx"):
                gx = _grad_x(layer, gyf, wpacked).view(ctx.xshape)
            return gx.to(gy.dtype), None, None
        if layer.int8:
            xq, xs, wpacked = ctx.saved_tensors
            with _rf("tern.dw"):
                gw = gyf.transpose(0, 1) @ (xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype))
        else:
            x, wpacked = ctx.saved_tensors
            xq = _act_quant_plain(x, layer.act_bits).reshape(-1, layer.K)
            xs = None
            gw = gyf.transpose(0, 1) @ xq                       # [N,K] dense grad

        with _rf("tern.dx"):
            gx = _grad_x(layer, gyf, wpacked).view(ctx.xshape)
        if getattr(layer, "cap_accum", False) and getattr(layer, "ts_micro", False):
            # --ttf-accum micro: the TTF update sees every micro-batch's gradient (momentum and v step per micro-batch; the
            # accumulator, firing and spend on the last one), so no dense gradient is kept across micro-batches
            layer._ts_final = getattr(layer, "cap_final", True)
            layer.on_grad(layer, gw.detach().float())
            del gw
            return gx.to(gy.dtype), None, None
        if getattr(layer, "cap_accum", False):
            # gradient accumulation with a captured gradient (TTF, grad_accum > 1): sum the micro-batches (each
            # loss is already divided by grad_accum); with --ttf-update fused the update runs on the last micro-batch only
            g = gw.detach().float() if layer.gw is None else layer.gw + gw.float()
            del gw
            if getattr(layer, "on_grad", None) is not None and getattr(layer, "cap_final", True):
                layer.gw = None
                layer.on_grad(layer, g)
            else:
                layer.gw = g
            return gx.to(gy.dtype), None, None
        if getattr(layer, "on_grad", None) is not None:
            # --ttf-update fused: the layer's update runs here, after its input gradient (gx) is computed, so this layer's
            # trits are not needed again in this step (earlier layers recompute with their own, unchanged trits)
            layer.on_grad(layer, gw.detach().float())
            del gw
            return gx.to(gy.dtype), None, None
        layer.gw = gw.detach().float()          # the update after the backward (ts_step without --ttf-update fused), diagnostics
        if getattr(layer, "capture_stats", False):
            # per-input-channel and per-output-row RMS, for selection diagnostics
            xf2 = (xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype)) if layer.int8 else xq
            layer.x_rms = xf2.float().pow(2).mean(0).sqrt().detach()
            layer.gy_rms = gyf.float().pow(2).mean(0).sqrt().detach()
        if getattr(layer, "capture_h", False):
            # Gauss-Newton diagonal for this layer: H[n,k] = sum_m (beta*gy)^2 x^2.
            # Same shape and cost as the weight gradient, one extra GEMM.
            xf = (xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype)) if layer.int8 else xq
            layer.hw = ((gyf * gyf).transpose(0, 1) @ (xf * xf)).detach().float()
        return gx.to(gy.dtype), None, None


def _grad_x(layer, gyf, wpacked):
    if layer.int8 and layer.int8_dx:
        return tern_gemm_dx_i8(gyf, wpacked, layer.K)
    return tern_gemm_dx(gyf, wpacked, layer.K)


class KernelTernaryLinear(nn.Module):
    """Pure-ternary linear, forward on the Triton packed-ternary GEMM (weights
    stay packed, never materialized dense). Row-packed 5-trits/byte along K."""
    def __init__(self, in_features, out_features, act_bits: int = 8, beta: bool = False, int8: bool = False,
                 int8_dx: bool = False):
        super().__init__()
        self.use_beta = beta      # scale output by 1/sqrt(K*rho); False = legacy raw trits
        self.int8 = int8          # s8 tensor-core path (fwd exact, dx 8-bit gy)
        self.int8_dx = int8_dx    # use the s8 dx kernel (else the bf16 one)
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
            self._beta_val = trit_beta(wpacked, self.K); self._beta_key = k
        return self._beta_val

    def _invalidate(self):
        self._beta_key = None

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
        unchanged at first) unless --adapter-init svd set it"""
        dev = self.wpacked.device
        self.mag_kind = kind
        assert kind == "add", "the adapter is additive (the multiplicative and ternary kinds are in ttf.legacy)"
        mi_ = getattr(self, "_mag_init", None)
        if mi_ is not None:      # --adapter-init svd: the rounding error's best rank-r part (hf_init._mag_svd_init)
            assert kind == "add" and mi_[0].shape[1] == r, "--adapter-init svd: the fitted rank must equal --adapter-rank"
            self.mag_A = nn.Parameter(mi_[0].to(dev).clone()); self.mag_B = nn.Parameter(mi_[1].to(dev).clone())
            del self._mag_init
            return [self.mag_A, self.mag_B]
        self.mag_A = nn.Parameter(torch.zeros(self.N, r, device=dev))
        std = (1.0 / self.K) ** 0.5
        self.mag_B = nn.Parameter(torch.randn(self.K, r, device=dev) * std)
        return [self.mag_A, self.mag_B]

    def enable_centering(self, mu, bias):
        """--convert gptqc: y = layer(x - mu) + bias, mu a fixed buffer, bias a float parameter (init W mu)"""
        self.register_buffer("in_mean", mu.detach().clone().float())
        self.out_bias = nn.Parameter(bias.detach().clone().float())

    def forward(self, x):
        if getattr(self, "in_mean", None) is not None:      # gptqc: centred input (before the activation quantisation)
            x = x - self.in_mean.to(x.dtype)
        # column scale before the GEMM (and before the int8 activation quantization),
        # row scale after it: the packed kernels are unchanged, autograd handles r and c,
        # and the update's dL/dq comes out scaled by r_i * c_j per weight
        fz = (FUSED_SCALES[0] and x.is_cuda and x.dtype == torch.bfloat16 and self.col_scale is not None
              and self.row_scale is not None and getattr(self, "mag_A", None) is not None)
        if fz:      # --fused-scales: the same layer, the elementwise work in fewer passes
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

    @torch.no_grad()
    def ternary_weight(self):
        w = unpack_rows(self.wpacked, self.K)
        if self.use_beta:
            return w, trit_beta(self.wpacked, self.K).item()
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale


def build_kernel_transformer(c: ModelConfig, grad_checkpoint: bool = True, beta: bool = False, int8: bool = False,
                             int8_dx: bool = False):
    def make_linear(i, o):
        return KernelTernaryLinear(i, o, c.act_bits, beta=beta, int8=int8, int8_dx=int8_dx)
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
