"""Ternary weights trained by a learned FLIP PREDICTOR.

No latent weight, no int8 master. Each weight is a pure trit w in {-1,0,1}.
Training never nudges a continuous value; instead a small shared network — the
*flip predictor* — looks at the backprop signal and predicts the discrete
transition for each weight: shift down (e.g. 0 -> -1), stay, or shift up
(e.g. 0 -> +1). Only whole-level flips ever happen.

Mechanism per BitLinear-shaped layer
------------------------------------
State (buffers only, no nn.Parameter):
  weight   int8 in {-1,0,1}          -> packs to ~1.58 bit at deploy
  evidence int8                       -> integrate-and-fire accumulator of grad

Forward:
  feat   = [weight, evidence]                     (per element)
  logits = predictor(feat)  -> P(down), P(stay), P(up)
  delta  = P(up) - P(down)  in [-1,1]             (predicted flip direction)
  w_used = weight  (value), with a straight-through path so the task loss
           trains the predictor: grad to delta == step * dL/dweight.
  y = ternary_matmul(quantized_x, w_used)

Backward (hook on w_used):
  g = dL/dweight. Descent wants weight to move opposite g, so
  evidence += quantize(-g). (This is the accumulated backprop pressure.)

Flip step (called once per optimizer step, after backward):
  A weight flips when the integrated evidence crosses a threshold AND the
  predictor agrees on the direction:
     evidence >= +theta and predictor delta > 0 and weight < 1  -> weight += 1
     evidence <= -theta and predictor delta < 0 and weight > -1 -> weight -= 1
  evidence resets on flip. The predictor is the learned veto/accelerator that
  decides whether a given backprop step should actually shift the bit.

The predictor is ONE tiny shared network (~a few hundred params) trained by the
task loss for the whole model. Persistent per-weight memory: weight + evidence.
Set evidence_bits low, or disable the predictor veto, to trade stability for
memory.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function as _rf      # named ranges for TERN_PROF (cheap when not profiling)

from .config import ModelConfig
from .model import BitTransformer
from .bitlinear import _act_quant_ste
from .packed import TritStore, unpack5, pack5


def _act_quant_plain(x, bits=8):
    """Detached 8-bit activation quant (no autograd graph); used inside the
    custom Function where the straight-through path is handled manually."""
    qp = 2 ** (bits - 1) - 1
    scale = qp / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (x * scale).round().clamp_(-qp - 1, qp) / scale


class _PackedStatelessFn(torch.autograd.Function):
    """Memory-bounded ternary linear. Forward saves only the input activation and
    the packed weight (tiny) — NOT the dense bf16 weight. Backward re-unpacks
    transiently, computes the weight gradient, applies the stochastic flip to the
    packed buffer in place, then frees everything. Peak weight-space memory is one
    layer, independent of depth."""
    @staticmethod
    def forward(ctx, x, packed, layer):
        w = unpack5(packed, layer.numel).view(layer.shape).to(x.dtype)
        xq = _act_quant_plain(x, layer.act_bits)
        y = F.linear(xq, w)
        ctx.layer = layer
        ctx.save_for_backward(x, packed)
        del w
        return y

    @staticmethod
    def backward(ctx, gy):
        x, packed = ctx.saved_tensors
        layer = ctx.layer
        w = unpack5(packed, layer.numel).view(layer.shape).to(gy.dtype)
        gx = torch.matmul(gy, w)                      # STE through act-quant
        del w
        xq = _act_quant_plain(x, layer.act_bits)
        gw = torch.matmul(gy.reshape(-1, gy.shape[-1]).transpose(0, 1),
                          xq.reshape(-1, xq.shape[-1]))   # [out, in]
        layer._flip_from_grad(gw, packed)             # mutate packed in place
        del gw
        return gx, None, None


class FlipPredictor(nn.Module):
    """Shared net: per-weight features -> {down, stay, up} logits."""
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        # bias 'stay' initially so an untrained predictor is cautious
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, 1.0, 0.0]))

    def forward(self, feat):          # feat [..., 2] -> [..., 3]
        return self.net(feat)


class TernaryFlipLinear(nn.Module):
    def __init__(self, in_features, out_features, predictor: FlipPredictor,
                 act_bits: int = 8, theta: float = 24.0, e_gain: float = 4.0,
                 step: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.theta = theta          # evidence needed to fire a flip
        self.e_gain = e_gain        # how fast evidence integrates grad
        self.step = step            # STE scale that trains the predictor
        # wrap predictor in a list so it is NOT registered as a submodule here
        # (it is owned once by the top model, trained once).
        self._pred = [predictor]

        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        scale = w.abs().mean().clamp_min(1e-5)
        weight = (w / scale).round().clamp_(-1, 1).to(torch.int8)   # {-1,0,1}
        # weight stored packed 5-trits/byte (~1.6 bit); evidence kept int8
        self.wstore = TritStore(self, "weight_packed", weight)
        self.register_buffer("evidence", torch.zeros_like(weight))
        self._delta = None          # last predicted flip direction (detached)

    @property
    def predictor(self):
        return self._pred[0]

    def forward(self, x):
        wf = self.wstore.unpack().to(x.dtype)           # transient unpack
        ef = self.evidence.to(x.dtype) / 127.0
        feat = torch.stack([wf, ef], dim=-1)            # [out,in,2]
        logits = self.predictor(feat)
        p = logits.softmax(dim=-1)
        delta = p[..., 2] - p[..., 0]                   # [-1,1]

        # value == pure ternary weight; gradient flows to predictor via delta
        w_used = wf + self.step * delta - (self.step * delta).detach()

        x_q = _act_quant_ste(x, self.act_bits)
        y = F.linear(x_q, w_used)

        if w_used.requires_grad:
            w_used.register_hook(self._integrate)
        self._delta = delta.detach()
        return y

    def _integrate(self, g):
        """Accumulate backprop pressure into the int8 evidence register."""
        with torch.no_grad():
            gn = g / g.detach().abs().mean().clamp_min(1e-8)   # normalize scale
            inc = torch.round(-gn * self.e_gain)               # descent direction
            e = self.evidence.to(torch.float32) + inc
            self.evidence.copy_(e.clamp_(-127, 127).to(torch.int8))
        return g   # pass gradient through unchanged

    @torch.no_grad()
    def flip_step(self):
        """Apply predicted flips. Returns number of weights flipped."""
        if self._delta is None:
            return 0
        e = self.evidence.to(torch.float32)
        w = self.wstore.unpack()
        d = self._delta
        up = (e >= self.theta) & (d > 0) & (w < 1)
        dn = (e <= -self.theta) & (d < 0) & (w > -1)
        w[up] += 1
        w[dn] -= 1
        fired = up | dn
        if fired.any():
            self.wstore.write(w)                         # repack changed trits
        self.evidence[fired] = 0
        return int(fired.sum().item())

    @torch.no_grad()
    def ternary_weight(self):
        """Deploy view: {-1,0,1} trits + one absmean scale."""
        w = self.wstore.unpack()
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"ternary flip-predicted, theta={self.theta}")


class StatelessFlipLinear(nn.Module):
    """Zero-accumulator ternary layer. Only per-weight state is the trit itself
    (packs to ~1.58 bit / 6.8 GiB at 37B). No evidence register, no predictor.

    The gradient noise is averaged in TIME instead of in a register: each step,
    each weight flips one level toward the descent direction with a small
    probability that scales with this step's gradient magnitude. Over many steps
    the expected drift follows the true gradient; a near-zero-gradient weight
    flips rarely and symmetrically, so it stays put on average.

    Honest: slower and noisier than the evidence version, convergence at LLM
    scale is unproven. This is the literal 'no accumulation at all' option.
    """
    def __init__(self, in_features, out_features, act_bits: int = 8,
                 rate: float = 2e-3, g_ref: float = 3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.rate = rate            # base flip probability ceiling
        self.g_ref = g_ref          # grad (in units of mean|g|) that saturates prob
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        scale = w.abs().mean().clamp_min(1e-5)
        weight = (w / scale).round().clamp_(-1, 1).to(torch.int8)
        # only state: weight, packed 5-trits/byte (~1.6 bit / 7.2 GB at 37B)
        self.wstore = TritStore(self, "weight_packed", weight)
        self.numel = weight.numel()
        self.shape = tuple(weight.shape)

    def forward(self, x):
        return _PackedStatelessFn.apply(x, self.weight_packed, self)

    @torch.no_grad()
    def _flip_from_grad(self, g, packed):
        """Stochastic flip applied straight into the packed buffer (in place)."""
        gn = g / g.abs().mean().clamp_min(1e-8)
        prob = (gn.abs() / self.g_ref).clamp_(0, 1) * self.rate
        fire = torch.rand_like(prob) < prob
        if not fire.any():
            return
        w = unpack5(packed, self.numel).view(self.shape)
        direction = -torch.sign(gn).to(torch.int8)                # descend
        w[fire] = (w + direction).clamp_(-1, 1)[fire]
        packed.copy_(pack5(w))

    @torch.no_grad()
    def ternary_weight(self):
        w = self.wstore.unpack()
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"stateless stochastic flip, rate={self.rate}")


def build_stateless_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                                rate: float = 2e-3):
    def make_linear(i, o):
        return StatelessFlipLinear(i, o, c.act_bits, rate=rate)
    return BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)


# ---- Triton-kernel stateless layer (fast path) ----------------------------
from .kernel import (pack_rows, unpack_rows, unpack_fast, tern_gemm, tern_gemm_dx, fused_flip,
                     fused_flip_ev, trit_beta, trit_diff, popcount,
                     tern_gemm_i8, tern_gemm_dx_i8, act_quant_i8, act_quant_i8_fast, dw_int8, dw_sign,
                     dw_cublas, expand_trit_mask, fused_flip_lock, fused_flip_row)

_FLIP_SEED = [0]


class _TernNoFlipFn(torch.autograd.Function):
    """beta * Q8(u) @ T^T with the layer's packed trits, input gradient only: no weight-gradient capture and no
    flips. Used by the low-rank multiplicative magnitude term (enable_lowrank_mag)."""
    @staticmethod
    def forward(ctx, u, wpacked, layer):
        beta = trit_beta(wpacked, layer.K) if layer.use_beta else None
        xq, xs = act_quant_i8(u.reshape(-1, layer.K), layer.act_bits)
        y = tern_gemm_i8(xq, xs, wpacked, layer.K, beta if beta is not None else 1.0)
        ctx.save_for_backward(wpacked); ctx.layer = layer; ctx.beta = beta; ctx.ushape = u.shape
        return y.view(*u.shape[:-1], layer.N).to(u.dtype)

    @staticmethod
    def backward(ctx, gy):
        (wpacked,) = ctx.saved_tensors
        layer = ctx.layer
        w = unpack_rows(wpacked, layer.K).to(gy.dtype)            # dense [N, K], transient
        g = gy.reshape(-1, layer.N)
        if ctx.beta is not None:
            g = g * ctx.beta.to(g.dtype)
        gu = (g @ w).view(ctx.ushape)                             # STE through the activation quantization
        return gu, None, None


class _TernBoostFn(torch.autograd.Function):
    """--ts_boost: beta(T) * Q8(x) @ B^T with the layer's packed boost plane B, input gradient only (TTF moves B itself)"""
    @staticmethod
    def forward(ctx, x, bpacked, layer):
        beta = trit_beta(layer.wpacked, layer.K) if layer.use_beta else None     # the main trits' beta
        xq, xs = act_quant_i8(x.reshape(-1, layer.K), layer.act_bits)
        y = tern_gemm_i8(xq, xs, bpacked, layer.K, beta if beta is not None else 1.0)
        ctx.save_for_backward(bpacked); ctx.layer = layer; ctx.beta = beta; ctx.xshape = x.shape
        return y.view(*x.shape[:-1], layer.N).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        (bpacked,) = ctx.saved_tensors
        layer = ctx.layer
        w = unpack_rows(bpacked, layer.K).to(gy.dtype)
        g = gy.reshape(-1, layer.N)
        if ctx.beta is not None:
            g = g * ctx.beta.to(g.dtype)
        return (g @ w).view(ctx.xshape), None, None


def enable_boost(model, s0=0.5):
    """--ts_boost S: every ternary layer gets a second, initially empty, ternary plane B and a learnable float scale s
    (init S): W = T + s B. TTF writes B when a trit at +-1 still has evidence past the threshold in its own direction, and
    clears it first when evidence comes back (see ts_layer). 0.2 bytes per weight (packed), one more ternary GEMM."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.register_buffer("bpacked", torch.zeros_like(m.wpacked) + pack_rows(torch.zeros(1, m.K, dtype=torch.int8,
                              device=m.wpacked.device)))
            m.boost_s = nn.Parameter(torch.tensor(float(s0), device=m.wpacked.device))
            n += 1
    return n


FACTORED = [False]           # --ts_factored: on for this step (off on --ts_vdiag / --ts_diag steps, which read dense g)
DW_FP32 = [False]            # --dw_fp32: dense weight gradients in fp32 from the GEMM (more precise, one copy fewer)
WCACHE = {"mode": "off"}     # --w_cache int8: keep each layer's unpacked int8 trits between updates (1 byte / weight)


class _KernelTernFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wpacked, layer):
        beta = layer._cached_beta(wpacked) if layer.use_beta else None
        if layer.int8:
            # int8 path: activations are already 8-bit, so x @ W^T is exact in int32
            # on the s8 tensor cores, and the saved activation is 1 byte/element.
            xq, xs = (act_quant_i8_fast if x.is_cuda else act_quant_i8)(x.reshape(-1, layer.K), layer.act_bits)
            with _rf("tern.fwd"):
                y = tern_gemm_i8(xq, xs, wpacked, layer.K, beta if beta is not None else 1.0,
                                 w8=layer._cached_w8(wpacked))
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
                and not layer.err_feedback and layer.dw_mode == "dense"
                and (getattr(layer, "ts_micro", False) or not getattr(layer, "cap_accum", False))):
            # --ts_factored: no dense weight gradient; the TTF update gets its factors (gy, x), g = gy^T x
            xq, xs, wpacked = ctx.saved_tensors
            with _rf("tern.dx"):
                gx = _grad_x(layer, gyf, wpacked).view(ctx.xshape)
            if getattr(layer, "cap_accum", False):
                layer._ts_final = getattr(layer, "cap_final", True)
            layer.on_grad(layer, (gyf, xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype)))
            return gx.to(gy.dtype), None, None
        # weight gradient first: error feedback needs it (and the flip) BEFORE grad_x
        if layer.int8:
            xq, xs, wpacked = ctx.saved_tensors
            if layer.dw_mode == "sign":
                gw = dw_sign(gyf, xq)
            elif layer.dw_mode == "int8":
                gw = dw_int8(gyf, xq, xs)
            elif layer.dw_mode == "cublas":
                gw = dw_cublas(gyf, xq, xs)
            else:
                with _rf("tern.dw"):
                    if DW_FP32[0]:   # --dw_fp32: the weight gradient straight out of the GEMM in fp32 (no bf16 rounding
                        gw = torch.mm(gyf.transpose(0, 1), xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype),   # and no copy)
                                      out_dtype=torch.float32)
                    else:
                        gw = gyf.transpose(0, 1) @ (xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype))
        else:
            x, wpacked = ctx.saved_tensors
            xq = _act_quant_plain(x, layer.act_bits).reshape(-1, layer.K)
            xs = None
            gw = gyf.transpose(0, 1) @ xq                       # [N,K] dense grad

        if layer.err_feedback and not layer.capture:
            # ---- spatial error feedback -------------------------------------
            # The flip rule samples a discrete move: E[dw] = p * d, but what is
            # applied is fired * d (d = -sign(g)). The difference is the demand this
            # layer failed to satisfy this step -- dominated NOT by the weights that
            # stayed put (p is tiny) but by the ones that fired and so moved a full
            # level instead of p of one. It is pushed to the layers behind us in the
            # same backward pass (a change in x can produce the change in y the
            # weight change would have), so nothing is stored across steps.
            gx = _ef_backward(layer, gyf, xq, xs, gw, wpacked)
            del gw
            return gx.view(ctx.xshape).to(gy.dtype), None, None

        with _rf("tern.dx"):
            gx = _grad_x(layer, gyf, wpacked).view(ctx.xshape)
        if layer.capture and getattr(layer, "cap_accum", False) and getattr(layer, "ts_micro", False):
            # --ts_micro: the TTF update sees every micro-batch's gradient (momentum and v step per micro-batch; the
            # accumulator, firing and spend on the last one), so no dense gradient is kept across micro-batches
            layer._ts_final = getattr(layer, "cap_final", True)
            layer.on_grad(layer, gw.detach().float())
            del gw
            return gx.to(gy.dtype), None, None
        if layer.capture and getattr(layer, "cap_accum", False):
            # gradient accumulation with a captured gradient (--ts, grad_accum > 1): sum the micro-batches (each
            # loss is already divided by grad_accum); with --ts_fused the update runs on the last micro-batch only
            g = gw.detach().float() if layer.gw is None else layer.gw + gw.float()
            del gw
            if getattr(layer, "on_grad", None) is not None and getattr(layer, "cap_final", True):
                layer.gw = None
                layer.on_grad(layer, g)
            else:
                layer.gw = g
            return gx.to(gy.dtype), None, None
        if layer.capture and getattr(layer, "on_grad", None) is not None:
            # --ts_fused: the layer's update runs here, after its input gradient (gx) is computed, so this layer's
            # trits are not needed again in this step (earlier layers recompute with their own, unchanged trits)
            layer.on_grad(layer, gw.detach().float())
            del gw
            return gx.to(gy.dtype), None, None
        if layer.capture:
            layer.gw = gw.detach().float()      # diagnostics (diag_snr.py): keep grad, no flip
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
        if layer.accum_steps > 1:
            # gradient accumulation: sum dL/dW over micro-steps and flip ONCE per
            # optimizer step (flip_accumulated), so "tokens per step" means the same
            # thing for the flip rule as it does for the tail optimizer.
            if layer.gw_accum is None:
                layer.gw_accum = torch.zeros(layer.N, layer.K, dtype=torch.float32,
                                             device=gw.device)
            layer.gw_accum += gw.float()
            layer.accum_seen += 1
            del gw
            return gx.to(gy.dtype), None, None
        _FLIP_SEED[0] += 1
        before = wpacked.clone() if layer.track else None
        if layer.norm_mode == "row":
            fused_flip_row(wpacked, gw, layer.rate, layer.g_ref, _FLIP_SEED[0])
            layer._invalidate()
            if layer.track:
                layer._record_flips(before, wpacked)
            del gw
            return gx.to(gy.dtype), None, None
        gm = layer._flip_scale(gw)
        if layer.lockout > 0:
            fused_flip_lock(wpacked, gw, layer.lock, layer.lock_dir, layer.rate,
                            layer.g_ref, _FLIP_SEED[0], gmean=gm,
                            mode=layer.lockout_mode)
            layer._invalidate()
        elif layer.evidence is not None:
            fused_flip_ev(wpacked, gw, layer.evidence, layer.rate, layer.g_ref,
                          _FLIP_SEED[0], smax=layer.ev_max, gmean=gm)
            layer._invalidate()
        else:
            fused_flip(wpacked, gw, layer.rate, layer.g_ref, _FLIP_SEED[0], gmean=gm,
                       inv=layer.inv_prob, hump=layer.hump, hump_g0=layer.hump_g0,
                       hump_alpha=layer.hump_alpha)
            layer._invalidate()
        if before is not None:
            layer._record_flips(before, wpacked)
        del gw
        return gx.to(gy.dtype), None, None


def _grad_x(layer, gyf, wpacked):
    if layer.int8 and layer.int8_dx:
        return tern_gemm_dx_i8(gyf, wpacked, layer.K)
    return tern_gemm_dx(gyf, wpacked, layer.K, w8=layer._cached_w8(wpacked))


@torch.no_grad()
def _ef_backward(layer, gyf, xq, xs, gw, wpacked):
    """Flip first, measure what the flip actually did, fold the unmet demand into
    gy, then compute grad_x once from the corrected gy. Returns grad_x [M, K].

    Only valid for the plain stochastic flip (kernel mode, no evidence counter, no
    gradient accumulation): p below is that rule's exact flip probability.
    """
    gm = layer._flip_scale(gw)          # the denominator the kernel will use (live or frozen)
    gn = gw.float() / gm
    d = -torch.sign(gn)                                     # descent direction
    p = (gn.abs() / layer.g_ref).clamp_(0, 1) * layer.rate  # P(flip) per weight
    del gn

    before = wpacked.clone()
    _FLIP_SEED[0] += 1
    fused_flip(wpacked, gw, layer.rate, layer.g_ref, _FLIP_SEED[0], gmean=gm)
    layer._invalidate()
    fired = expand_trit_mask(trit_diff(before, wpacked), layer.K)
    if layer.track:
        layer._record_flips(before, wpacked)

    # residual demand in weight space, mapped into y space: r = x @ R^T
    R = ((p - fired.to(p.dtype)) * d).to(gyf.dtype)         # [N, K]
    del p, d, fired
    xf = xq.to(gyf.dtype) if xs is None else xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype)
    r = xf @ R.transpose(0, 1)                               # [M, N]
    del R, xf
    # r is the output change this layer still OWES (a desired delta-y). Descending
    # gy moves y by -gy, so asking earlier layers to deliver +r means SUBTRACTING
    # it from gy. (Adding it asks them to amplify the overshoot instead.)
    # Scale-match to gy: the flip rule normalizes by mean|g|, so only the SHAPE of
    # the added signal matters and alpha is a dimensionless mixing weight.
    r = r * (gyf.abs().mean() / r.abs().mean().clamp_min(1e-12))
    gyf = gyf - layer.ef_alpha * r
    del r
    # grad_x must be taken at the weights the FORWARD used, i.e. before this flip
    return _grad_x(layer, gyf, before)


class KernelTernaryLinear(nn.Module):
    """Pure-ternary linear, forward on the Triton packed-ternary GEMM (weights
    stay packed, never materialized dense). Row-packed 5-trits/byte along K."""
    def __init__(self, in_features, out_features, act_bits: int = 8,
                 rate: float = 2e-3, g_ref: float = 3.0, evidence: bool = False,
                 ev_bits: int = 2, beta: bool = False, int8: bool = False,
                 dw_mode: str = "int8", int8_dx: bool = False):
        super().__init__()
        self.use_beta = beta      # scale output by 1/sqrt(K*rho); False = legacy raw trits
        self.int8 = int8          # s8 tensor-core path (fwd exact, dx 8-bit gy)
        self.int8_dx = int8_dx    # use the s8 dx kernel (else the bf16 one)
        self.dw_mode = dw_mode    # int8 | cublas | sign (XNOR) | dense
        self.K = in_features
        self.N = out_features
        self.act_bits = act_bits
        self.rate = rate
        self.g_ref = g_ref
        self.norm_mode = "tensor"   # "tensor" = one mean|g| per layer; "row" = per output unit
        self.inv_prob = 0           # 1 = flip the SMALLEST gradients first
        self.hump = 0               # 1 = probability ~ measured net gain per flip
        self.hump_g0 = 37.0
        self.hump_alpha = 1.8
        self.ev_max = 2 ** (ev_bits - 1) - 1      # 2-bit->1, 3-bit->3, 4-bit->7
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        scale = w.abs().mean().clamp_min(1e-5)
        weight = (w / scale).round().clamp_(-1, 1).to(torch.int8)
        self.register_buffer("wpacked", pack_rows(weight))       # [N, ceil(K/5)]
        # 3-level (2-bit) per-weight counter, {-1,0,1}; int8 storage at this scale
        if evidence:
            self.register_buffer("evidence",
                                 torch.zeros(out_features, in_features, dtype=torch.int8))
        else:
            self.evidence = None
        # --- optional learned per-row / per-column scales (enable_rc_scale) -------
        # effective weight W_ij = beta * r_i * c_j * q_ij, from N + K floats per layer
        self.row_scale = None
        self.col_scale = None
        # --- diagnostics: capture dL/dW instead of flipping (see diag_snr.py) ---
        self.capture = False
        self.capture_h = False
        self.gw = None
        self.hw = None
        # --- gradient accumulation for the flip rule ----------------------------
        self.accum_steps = 1      # micro-steps per optimizer step
        self.accum_seen = 0
        self.gw_accum = None
        # --- flip threshold scale (Arm B: frozen absolute scale) ----------------
        # abs_scale=False -> denominator is this step's mean|g| (relative, cannot
        # converge: if every gradient shrinks, the ratio is unchanged and the same
        # fraction keeps flipping). True -> average mean|g| over the first
        # calib_steps then FREEZE, so the flip rate falls on its own.
        self.abs_scale = False
        self.calib_steps = 200
        self.calib_sum = 0.0
        self.calib_n = 0
        self.last_gmean = 0.0
        self.register_buffer("frozen_scale", torch.zeros(()), persistent=True)
        # --- spatial error feedback (no persistent state; see _ef_backward) -----
        self.err_feedback = False
        self.ef_alpha = 1.0
        # --- per-weight flip lockout (1 bit/weight in each of two packed masks) -
        # lockout > 0: a weight may flip at most once per `lockout` steps ("once"),
        # or after its first flip may only continue in the same direction
        # ("noreversal"). Masks are cleared by the trainer each epoch.
        self.lockout = 0
        self.lockout_mode = 0     # 0 = once, 1 = noreversal
        self.lock = None
        self.lock_dir = None
        # --- flip instrumentation (off unless enable_tracking() is called) ------
        self.track = False
        self.touched = None      # uint8 [N,K5] sticky mask: trit ever changed
        self.flips = None        # 0-dim int64: trit changes since last reset

    # --w_cache: the trits change only when an update writes wpacked (copy_ bumps its version; the Triton flip kernels
    # call _invalidate), so beta and the unpacked int8 weight are computed once per change instead of on every
    # forward, backward and recompute. Same values, bit for bit.
    def _wkey(self, wpacked):
        return (wpacked.data_ptr(), wpacked._version)

    @torch.no_grad()
    def _cached_beta(self, wpacked):
        k = self._wkey(wpacked)
        if getattr(self, "_beta_key", None) != k:
            self._beta_val = trit_beta(wpacked, self.K); self._beta_key = k
        return self._beta_val

    @torch.no_grad()
    def _cached_w8(self, wpacked):
        if WCACHE["mode"] != "int8":
            return None
        k = self._wkey(wpacked)
        if getattr(self, "_w8_key", None) != k:
            self._w8 = None
            self._w8 = unpack_fast(wpacked, self.K); self._w8_key = k
        return self._w8

    def _invalidate(self):
        self._beta_key = None; self._w8_key = None; self._w8 = None

    @torch.no_grad()
    def _flip_scale(self, grad_w):
        """Denominator for the flip threshold; see abs_scale."""
        g = grad_w.abs().mean().clamp_min(1e-8).item()
        self.last_gmean = g
        if not self.abs_scale:
            return g
        if float(self.frozen_scale) > 0.0:
            return float(self.frozen_scale)
        self.calib_sum += g
        self.calib_n += 1
        if self.calib_n >= self.calib_steps:
            self.frozen_scale.fill_(self.calib_sum / self.calib_n)
        return g

    def enable_lockout(self, steps: int, mode: str = "once"):
        self.lockout = int(steps)
        self.lockout_mode = 0 if mode == "once" else 1
        self.lock = torch.zeros_like(self.wpacked)
        self.lock_dir = torch.zeros_like(self.wpacked)

    @torch.no_grad()
    def reset_lock(self):
        if self.lock is not None:
            self.lock.zero_(); self.lock_dir.zero_()

    @torch.no_grad()
    def locked_frac(self):
        """Fraction of weights that have already flipped in this epoch."""
        if self.lock is None:
            return torch.zeros((), dtype=torch.int64, device=self.wpacked.device)
        return popcount(self.lock)

    def enable_tracking(self):
        """Track per-step flip count and which trits have ever changed since init.

        Costs one packed-size uint8 mask (~1.6 bit/weight) plus a packed clone per
        backward. Counts trit CHANGES per fused_flip call, so with grad_accum > 1
        the per-step number sums the micro-steps.
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
        if getattr(self, "rev_track", False):
            from .kernel import unpack_rows
            ta = unpack_rows(after, self.K).to(torch.int8)
            dt = ta - unpack_rows(before, self.K).to(torch.int8)
            ch = dt != 0
            # a reversal undoes this weight's previous change (back toward where it came from)
            self.revs += (ch & (dt == -self.last_dir)).sum()
            self.last_dir = torch.where(ch, dt, self.last_dir)
            self.net = (ta != self.t_start).sum()

    def enable_reversal_tracking(self):
        """Count flips that undo the weight's previous change, and the net displacement
        (trits that differ from the start of tracking). Two int8 [N, K] buffers."""
        from .kernel import unpack_rows
        self.rev_track = True
        self.t_start = unpack_rows(self.wpacked, self.K).to(torch.int8).clone()
        self.last_dir = torch.zeros_like(self.t_start)
        self.revs = torch.zeros((), dtype=torch.int64, device=self.wpacked.device)
        self.net = torch.zeros((), dtype=torch.int64, device=self.wpacked.device)

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
        """Low-rank float magnitude beyond the trits (A starts at 0, so the layer is unchanged at first):
        'add': W = beta*T + A B^T (a float adapter); 'mul': W = beta * T o (1 + A B^T), computed as
        T x + sum_k a_k o T (b_k o x) with r extra ternary GEMMs. r (N + K) floats."""
        dev = self.wpacked.device
        self.mag_kind = kind
        if kind == "taddw":   # the 'add' adapter in trits with the bytes of fp32 (rank 20 r: 1.6 bits vs 32 per entry):
            from .tern_emb import TernEmbed     # A_t [N, 20r], B_t [K, 20r], each s0 r c T, trained by TTF like the
            R = 20 * r                          # ternary embedding; A_t's row scales start at 0 (layer unchanged)
            self.mag_At = TernEmbed(self.N, R, "kernel", std=1.0).to(dev)
            self.mag_Bt = TernEmbed(self.K, R, "kernel", std=(1.0 / self.K) ** 0.5).to(dev)
            self.mag_At.row_scale.data.zero_()
            n_ = KernelTernaryLinear._taddw_n = getattr(KernelTernaryLinear, "_taddw_n", 0) + 1
            self.mag_At._ts_idx, self.mag_Bt._ts_idx = 20_000 + 2 * n_, 20_001 + 2 * n_   # TTF seeds, distinct per matrix
            return [self.mag_At.row_scale, self.mag_At.col_scale, self.mag_Bt.row_scale, self.mag_Bt.col_scale]
        mi_ = getattr(self, "_mag_init", None)
        if mi_ is not None:      # --init_mag_svd: the rounding error's best rank-r part (hf_init._mag_svd_init)
            assert kind == "add" and mi_[0].shape[1] == r, "--init_mag_svd R needs --lowrank_mag add:R"
            self.mag_A = nn.Parameter(mi_[0].to(dev).clone()); self.mag_B = nn.Parameter(mi_[1].to(dev).clone())
            del self._mag_init
            return [self.mag_A, self.mag_B]
        self.mag_A = nn.Parameter(torch.zeros(self.N, r, device=dev))
        std = (1.0 / self.K) ** 0.5 if kind == "add" else 1.0
        self.mag_B = nn.Parameter(torch.randn(self.K, r, device=dev) * std)
        return [self.mag_A, self.mag_B]

    def enable_res_anneal(self, R, rank=0):
        """--res_anneal: the fp weight's rounding residual R = W - W_hat (fixed, not saved; full or rank-r factors),
        added to the output scaled by RES_ANNEAL["f"] (1 -> 0 over the anneal): the layer starts as the fp layer"""
        if rank and rank < min(R.shape):
            U, S_, Vh = torch.linalg.svd(R.float(), full_matrices=False)
            self.register_buffer("res_U", (U[:, :rank] * S_[:rank]).to(torch.bfloat16).contiguous(), persistent=False)
            self.register_buffer("res_V", Vh[:rank].T.to(torch.bfloat16).contiguous(), persistent=False)
        else:
            self.register_buffer("res_W", R.to(torch.bfloat16).contiguous(), persistent=False)

    def enable_centering(self, mu, bias):
        """--init_hf_method gptqc: y = layer(x - mu) + bias, mu a fixed buffer, bias a float parameter (init W mu)"""
        self.register_buffer("in_mean", mu.detach().clone().float())
        self.out_bias = nn.Parameter(bias.detach().clone().float())

    def forward(self, x):
        f_ = RES_ANNEAL["f"]
        if f_ > 0 and (getattr(self, "res_W", None) is not None or getattr(self, "res_U", None) is not None):
            if getattr(self, "res_W", None) is not None:   # --res_anneal: the fading fp residual, on the raw input
                r_ = F.linear(x, self.res_W.to(x.dtype))
            else:
                r_ = (x @ self.res_V.to(x.dtype)) @ self.res_U.to(x.dtype).T
            return self._forward_t(x) + f_ * r_
        return self._forward_t(x)

    def enable_rotation(self, Q):
        """--init_rot: y = layer(Q x), Q a fixed orthogonal buffer (the trits were rounded from W Q^T)"""
        self.register_buffer("rot_Q", Q.detach().clone().float())

    def _forward_t(self, x):
        if getattr(self, "rot_Q", None) is not None:        # --init_rot: rotated input (before centring and quantisation)
            x = x @ self.rot_Q.to(x.dtype).T
        if getattr(self, "in_mean", None) is not None:      # gptqc: centred input (before the activation quantisation)
            x = x - self.in_mean.to(x.dtype)
        # column scale before the GEMM (and before the int8 activation quantization),
        # row scale after it: the packed kernels are unchanged, autograd handles r and c,
        # and the flip rule's dL/dq comes out scaled by r_i * c_j per weight
        if self.col_scale is not None:
            x = x * self.col_scale.to(x.dtype)
        y = _KernelTernFn.apply(x, self.wpacked, self)
        if getattr(self, "bpacked", None) is not None:      # --ts_boost: W = T + s B
            y = y + self.boost_s.to(y.dtype) * _TernBoostFn.apply(x, self.bpacked, self)
        if getattr(self, "mag_At", None) is not None:      # --lowrank_mag taddw:R
            y = y + (x @ self.mag_Bt._compute().to(x.dtype)) @ self.mag_At._compute().to(y.dtype).T
        if getattr(self, "mag_A", None) is not None:
            A, B = self.mag_A.to(y.dtype), self.mag_B.to(x.dtype)
            if self.mag_kind == "add":
                a = (x @ B) @ A.T
                cap = getattr(self, "mag_cap", 0.0)
                if cap:     # --mag_cap: the adapter stays a correction, rms(a) <= cap * rms(trit path)
                    s = (cap * y.detach().float().pow(2).mean().sqrt()
                         / a.detach().float().pow(2).mean().sqrt().clamp_min(1e-12)).clamp(max=1.0)
                    a = a * s.to(a.dtype)
                y = y + a
            else:
                for k in range(A.shape[1]):
                    y = y + A[:, k] * _TernNoFlipFn.apply(x * B[:, k], self.wpacked, self)
        if self.row_scale is not None:
            y = y * self.row_scale.to(y.dtype)
        if getattr(self, "out_bias", None) is not None:
            y = y + self.out_bias.to(y.dtype)
        return y

    @torch.no_grad()
    def _flip_from_grad(self, g, wpacked):
        gn = g / g.abs().mean().clamp_min(1e-8)
        prob = (gn.abs() / self.g_ref).clamp_(0, 1) * self.rate
        fire = torch.rand_like(prob) < prob
        if not fire.any():
            return
        w = unpack_rows(wpacked, self.K)                         # [N,K]
        direction = -torch.sign(gn).to(torch.int8)
        w[fire] = (w + direction).clamp_(-1, 1)[fire]
        wpacked.copy_(pack_rows(w))

    @torch.no_grad()
    def ternary_weight(self):
        w = unpack_rows(self.wpacked, self.K)
        if self.use_beta:
            return w, trit_beta(self.wpacked, self.K).item()
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale


RES_ANNEAL = {"f": 0.0}   # --res_anneal: weight of the fp rounding residual in every layer's output (1 -> 0)


def build_kernel_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                             rate: float = 2e-3, evidence: bool = False,
                             ev_bits: int = 2, beta: bool = False,
                             int8: bool = False, dw_mode: str = "int8",
                             int8_dx: bool = False, g_ref: float = 3.0):
    def make_linear(i, o):
        return KernelTernaryLinear(i, o, c.act_bits, rate=rate, evidence=evidence,
                                   ev_bits=ev_bits, beta=beta, int8=int8,
                                   dw_mode=dw_mode, int8_dx=int8_dx, g_ref=g_ref)
    return BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)


def build_flip_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                           theta: float = 24.0, e_gain: float = 4.0,
                           step: float = 0.05, hidden: int = 16):
    """BitTransformer whose linears are flip-predicted ternary layers, sharing
    one predictor (owned by the model so it trains once)."""
    predictor = FlipPredictor(hidden)

    def make_linear(i, o):
        return TernaryFlipLinear(i, o, predictor, c.act_bits, theta, e_gain, step)

    model = BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)
    model.add_module("flip_predictor", predictor)   # single owner -> trains once
    return model, predictor


@torch.no_grad()
def flip_accumulated(model):
    """Apply one flip per ternary layer from the accumulated gradient. No-op for
    layers running at accum_steps == 1 (they flip inside backward)."""
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear) and m.accum_steps > 1 and m.accum_seen:
            g = m.gw_accum / m.accum_seen
            _FLIP_SEED[0] += 1
            before = m.wpacked.clone() if m.track else None
            gm = m._flip_scale(g)
            if m.lockout > 0:
                fused_flip_lock(m.wpacked, g, m.lock, m.lock_dir, m.rate, m.g_ref,
                                _FLIP_SEED[0], gmean=gm, mode=m.lockout_mode)
            elif m.evidence is not None:
                fused_flip_ev(m.wpacked, g, m.evidence, m.rate, m.g_ref,
                              _FLIP_SEED[0], smax=m.ev_max, gmean=gm)
            else:
                fused_flip(m.wpacked, g, m.rate, m.g_ref, _FLIP_SEED[0], gmean=gm)
            if before is not None:
                m._record_flips(before, m.wpacked)
            m.gw_accum.zero_()
            m.accum_seen = 0


def set_err_feedback(model, enabled: bool, alpha: float = 1.0):
    """Spatial error feedback: push each layer's unapplied flip demand into grad_x."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            if enabled and (m.evidence is not None or m.accum_steps > 1):
                raise ValueError("error feedback needs plain stochastic flips: "
                                 "kernel mode, no evidence counter, grad_accum 1")
            m.err_feedback = enabled; m.ef_alpha = alpha; n += 1
    return n


def set_lockout(model, steps: int, mode: str = "once"):
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            if m.evidence is not None:
                raise ValueError("lockout is for plain stochastic flips (no evidence)")
            m.enable_lockout(steps, mode); n += 1
    return n


@torch.no_grad()
def reset_lockout(model):
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.reset_lock()


@torch.no_grad()
def lockout_stats(model):
    """Fraction of weights that have flipped in the current epoch (one sync)."""
    tot, locked = 0, []
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear) and m.lock is not None:
            locked.append(m.locked_frac()); tot += m.N * m.K
    if not locked:
        return None
    return float(torch.stack(locked).sum().item()) / tot


def set_hump(model, g0: float, alpha: float):
    """Flip probability proportional to measured net gain: (g/g_ref)(1-(g/g0)^alpha)."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.hump = 1; m.hump_g0 = g0; m.hump_alpha = alpha; n += 1
    return n


def set_inv_prob(model, inv: bool):
    """Reverse the probability ramp: small gradients flip most often."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.inv_prob = 1 if inv else 0; n += 1
    return n


def set_norm_mode(model, mode: str):
    """'tensor' (default) or 'row': which mean|g| normalises the flip threshold."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            if mode == "row" and (m.evidence is not None or m.lockout > 0 or m.abs_scale):
                raise ValueError("row normalisation is implemented for the plain "
                                 "stochastic flip only")
            m.norm_mode = mode; n += 1
    return n


def enable_rc_scales(model):
    """Learned per-row and per-column scales on every kernel ternary layer."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.enable_rc_scale(); n += m.N + m.K
    return n


def set_flip_rate(model, rate: float):
    """Arm A: open-loop flip-rate schedule."""
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.rate = rate


def set_abs_scale(model, enabled: bool, calib_steps: int = 200):
    """Arm B: freeze the flip-threshold denominator after calibration."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.abs_scale = enabled; m.calib_steps = calib_steps; n += 1
    return n


def set_flip_accum(model, steps: int):
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.accum_steps = steps; n += 1
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
    gmeans, frozen = [], []
    for name, m in model.named_modules():
        if isinstance(m, KernelTernaryLinear) and m.track:
            f, nv = m.flip_stats()
            names.append(name); flips.append(f); never.append(nv)
            numels.append(m.N * m.K)
            gmeans.append(m.last_gmean); frozen.append(float(m.frozen_scale))
    if not names:
        return {}
    f = torch.stack(flips).cpu().tolist()          # single sync
    nv = torch.stack(never).cpu().tolist()
    if reset:
        for m in model.modules():
            if isinstance(m, KernelTernaryLinear) and m.track:
                m.flips.zero_()
    tot = sum(numels)
    rv = [m for m in model.modules() if isinstance(m, KernelTernaryLinear) and getattr(m, "rev_track", False)]
    extra = {}
    if rv:
        r = torch.stack([m.revs for m in rv]).sum().item()
        extra = {"rev_flips": r, "net_frac": torch.stack([m.net for m in rv]).sum().item() / tot}
        for m in rv:
            m.revs.zero_()
    return {**extra,
        "layers": names,
        "flip_frac": [fi / n for fi, n in zip(f, numels)],
        "never_frac": [ni / n for ni, n in zip(nv, numels)],
        "flip_frac_total": sum(f) / tot,
        "never_frac_total": sum(nv) / tot,
        "gmean_layers": gmeans,
        "frozen_scale_layers": frozen,
    }


def apply_flips(model) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, TernaryFlipLinear):
            total += m.flip_step()
    return total
