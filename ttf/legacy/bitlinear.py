"""Master-free ternary BitLinear.

The trainable weight is stored ONLY as an int8 latent grid (`latent`) plus a
per-row float scale (`row_scale`). There is no separate high-precision master
copy and no optimizer state tensors sitting in memory.

How the update happens without a master copy
--------------------------------------------
1. Forward materializes a *transient* float weight  w_real = latent * row_scale.
   It is quantized to ternary (BitNet b1.58 absmean) with a straight-through
   estimator, so gradients flow back to w_real as if the quantizer were identity.
2. w_real registers a backward hook. When its gradient arrives during backprop,
   the hook immediately applies the optimizer step to the int8 `latent` in place
   (stochastic rounding), then w_real is freed.
3. With gradient checkpointing on, each layer's w_real is (re)built during that
   layer's backward and released right after, so peak float-weight memory is a
   single matrix, never the whole model.

Result: persistent weight memory ~= 1 byte/param (int8 latent), optionally
+1 byte/param for int8 momentum. No fp32 master, no fp32 Adam states.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainState:
    """Shared, mutable knobs the fused backward-hook optimizer reads each step."""
    def __init__(self):
        self.lr = 3e-4
        self.weight_decay = 0.0
        self.momentum = 0.0        # 0 -> stateless SGD+SR; >0 -> Lion-style int8 momentum
        self.updates_enabled = True  # off during eval / grad-accum non-final micro-steps
        self.grad_scale = 1.0        # divide incoming grad (for grad accumulation averaging)


# one global state object; the trainer mutates it
STATE = TrainState()


def _stochastic_round(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest integer with probability = fractional part (unbiased)."""
    fl = torch.floor(x)
    prob = x - fl
    return fl + (torch.rand_like(x) < prob).to(x.dtype)


def _act_quant_ste(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token symmetric absmax activation quantization with straight-through."""
    qp = 2 ** (bits - 1) - 1
    scale = qp / x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    xq = (x * scale).round().clamp_(-qp - 1, qp) / scale
    return x + (xq - x).detach()


def _ternary_ste(w_real: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 absmean ternary {-1,0,1}*scale with straight-through."""
    scale = w_real.detach().abs().mean().clamp_min(1e-5)
    t = (w_real / scale).round().clamp_(-1, 1) * scale
    return w_real + (t - w_real).detach()


class BitLinear(nn.Module):
    """Ternary linear layer, no bias, master-free training.

    Storage: latent (int8) [out,in], row_scale (float32) [out], optional
    momentum (int8) [out,in]. At deploy you keep only the ternary sign of the
    latent + one absmean scale (see pack.py).
    """
    def __init__(self, in_features: int, out_features: int, act_bits: int = 8,
                 rescale_every: int = 200):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.rescale_every = rescale_every
        self._step = 0

        # --- initialize a real weight, then discard it into int8 latent ------
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, mean=0.0, std=1.0 / math.sqrt(in_features))
        row_scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        latent = torch.round(w / row_scale[:, None]).clamp_(-127, 127).to(torch.int8)

        self.register_buffer("latent", latent)
        self.register_buffer("row_scale", row_scale)
        self.register_buffer("momentum", torch.zeros_like(latent), persistent=False)

    # -- the fused optimizer step, run inside the backward hook ---------------
    def _apply_grad(self, grad_wreal: torch.Tensor):
        if not STATE.updates_enabled:
            return
        with torch.no_grad():
            g = grad_wreal.to(torch.float32)
            if STATE.grad_scale != 1.0:
                g = g / STATE.grad_scale
            rs = self.row_scale[:, None]
            real = self.latent.to(torch.float32) * rs        # current real weight

            if STATE.momentum > 0.0:
                # Lion-style: sign of interpolated momentum drives the step
                m = self.momentum.to(torch.float32) * rs
                beta = STATE.momentum
                c = beta * m + (1.0 - beta) * g
                step = torch.sign(c) * STATE.lr
                m = 0.99 * m + 0.01 * g                       # momentum EMA
                self.momentum.copy_(
                    _stochastic_round(m / rs).clamp_(-127, 127).to(torch.int8))
            else:
                step = g * STATE.lr                            # plain SGD

            if STATE.weight_decay > 0.0:
                real = real * (1.0 - STATE.lr * STATE.weight_decay)
            real = real - step

            self._step += 1
            if self._step % self.rescale_every == 0:
                # keep the int8 grid centered on the current weight magnitude
                new_rs = real.abs().amax(dim=1).clamp_min(1e-8) / 127.0
                self.row_scale.copy_(new_rs)
                rs = new_rs[:, None]

            self.latent.copy_(
                _stochastic_round(real / rs).clamp_(-127, 127).to(torch.int8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # transient float weight; the ONLY high-precision copy, one layer's worth
        w_real = self.latent.to(x.dtype) * self.row_scale[:, None]
        w_real.requires_grad_(True)
        if w_real.requires_grad:
            w_real.register_hook(self._apply_grad)
        w_t = _ternary_ste(w_real)
        x_q = _act_quant_ste(x, self.act_bits)
        return F.linear(x_q, w_t)

    # -- deployment: collapse latent to pure ternary + one scale --------------
    @torch.no_grad()
    def ternary_weight(self):
        w_real = self.latent.to(torch.float32) * self.row_scale[:, None]
        scale = w_real.abs().mean().clamp_min(1e-5)
        t = (w_real / scale).round().clamp_(-1, 1).to(torch.int8)  # {-1,0,1}
        return t, scale

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, ternary, master-free"
