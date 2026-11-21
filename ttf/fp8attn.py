"""--attn-fp8 (21 Nov): causal attention with the forward on FP8 tensor cores (sm_89+: e4m3 at 2x the bf16 rate).

Forward (one Triton kernel, flash-attention 2 order): per query tile, Q quantized to e4m3 with one scale per tile; per
key tile, K and V quantized with one scale per tile each; S = (Q8 K8^T) sq sk in fp32, the online softmax in fp32, P
(in [0, 1] after the running-max shift) cast to e4m3 unscaled, O += (P8 V8) sv. The log-sum-exp of the scaled scores
is kept (natural log, the flash-attention layout [B, H, T] fp32).
Backward: aten's flash-attention backward on the bf16 q, k, v with this forward's output and log-sum-exp (it recomputes
P in bf16 from q, k and the log-sum-exp): the gradient of bf16 attention taken at the FP8 forward's output, the
straight-through rule of the low-precision forward."""
import math

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, sm_scale, T,
                sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sob, soh, sot, H,
                D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0); bh = tl.program_id(1)
    b = bh // H; h = bh % H
    rm = pm * BM + tl.arange(0, BM); rn = tl.arange(0, BN); rd = tl.arange(0, D)
    qp = q_ptr + b * sqb + h * sqh
    kp = k_ptr + b * skb + h * skh
    vp = v_ptr + b * svb + h * svh
    q = tl.load(qp + rm[:, None] * sqt + rd[None, :], mask=rm[:, None] < T, other=0.0).to(tl.float32)
    sq = tl.maximum(tl.max(tl.abs(q)), 1e-12) / 448.0
    q8 = (q / sq).to(tl.float8e4nv)
    qk_scale = sm_scale * 1.4426950408889634        # (exp2)
    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    hi = tl.minimum((pm + 1) * BM, T)
    for n0 in range(0, hi, BN):
        cn = n0 + rn
        k = tl.load(kp + cn[:, None] * skt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float32)
        v = tl.load(vp + cn[:, None] * svt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float32)
        sk = tl.maximum(tl.max(tl.abs(k)), 1e-12) / 448.0
        sv = tl.maximum(tl.max(tl.abs(v)), 1e-12) / 448.0
        k8 = (k / sk).to(tl.float8e4nv)
        v8 = (v / sv).to(tl.float8e4nv)
        s = tl.dot(q8, tl.trans(k8)) * (sq * sk * qk_scale)
        if n0 + BN > pm * BM:     # (the diagonal tiles: causal mask)
            s = tl.where((rm[:, None] >= cn[None, :]) & (cn[None, :] < T), s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float8e4nv), v8) * sv
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + b * sob + h * soh + rm[:, None] * sot + rd[None, :], o.to(tl.bfloat16), mask=rm[:, None] < T)
    # lse in natural log of the scaled scores: (m (log2 units) + log2 l) * ln 2
    tl.store(lse_ptr + bh * T + rm, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, mask=rm < T)


CFG = [None]   # (BM, BN, warps, stages); None: the default below


def fwd(q, k, v, sm_scale=None):
    """q, k, v [B, H, T, D] bf16 (any strides with a unit last stride) -> (o [B, H, T, D] bf16, lse [B, H, T] fp32)"""
    B, H, T, D = q.shape
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    o = torch.empty(B, H, T, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
    BM, BN, nw, ns = CFG[0] or (64, 64, 4, 3)    # (t_fp8attn.py 21 Nov, 4070 Ti Super: B16 T2048 H12 D64)
    _fwd_kernel[(triton.cdiv(T, BM), B * H)](q, k, v, o, lse, sm_scale, T,
                                             q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                                             v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
                                             H, D=D, BM=BM, BN=BN, num_warps=nw, num_stages=ns)
    return o, lse


_PH = {}


def _philox(dev):
    if dev not in _PH:
        z = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16, device=dev)
        r = torch.ops.aten._scaled_dot_product_flash_attention(z, z, z, 0.0, True, False)
        _PH[dev] = (r[6], r[7])
    return _PH[dev]


class FP8AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        o, lse = fwd(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        T = q.shape[2]
        ps, po = _philox(q.device)
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            do.contiguous() if do.stride(-1) != 1 else do, q, k, v, o, lse, None, None, T, T, 0.0, True, ps, po)
        return dq, dk, dv


def attention(q, k, v):
    """causal attention, q k v [B, H, T, D] bf16 CUDA (GQA already expanded)"""
    return FP8AttnFn.apply(q, k, v)
