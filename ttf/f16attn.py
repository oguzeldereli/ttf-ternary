"""--attn-f16 (21 Nov 2026): causal flash attention whose forward is a Triton kernel (flash-attention 2 order) with
O += P V on fp16 MMA with fp16 accumulation per key tile (GeForce Ada: 2x the fp32-accumulate rate, f16mm), and whose
backward is aten's flash-attention backward fed this forward's O and log-sum-exp (exact: the backward recomputes P
from the scores and that lse; an own Triton backward with fp16-accumulated dV / dK / dQ was slower than aten's flash
kernel, 7.8 vs 6.5 ms at 110M shapes, and was dropped).

The scores S = Q K^T accumulate in fp32 (the softmax's exp needs them exact to fp32 sums of exact fp16 products). fp16
range: q, k, v are bf16 values converted exactly (|x| < 65504); P is in [0, 1]. Output: O bf16 and the log-sum-exp
(base 2, of the scaled scores) fp32 [B, H, T]. 110M (B16 T2048 H12 D64, 4070 Ti Super): forward 1.21 vs SDPA 1.41 ms."""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
F16ACC = [True]
TILES = {"fwd": (128, 64, 4, 2)}   # (BM, BN, warps, stages; sweep 21 Nov, 4070 Ti Super)


@triton.jit
def _dot16(a, b, acc, F16: tl.constexpr):
    if F16:
        return acc + tl.dot(a, b, out_dtype=tl.float16).to(tl.float32)
    return tl.dot(a, b, acc)


@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, qk_scale, T, H, sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sgb, sgh, sgt, sob, soh, sot,
                D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, F16: tl.constexpr):
    pm = tl.program_id(0); bh = tl.program_id(1); b = bh // H; h = bh % H
    rm = pm * BM + tl.arange(0, BM); rn = tl.arange(0, BN); rd = tl.arange(0, D)
    offq = b * sqb + h * sqh; offk = b * skb + h * skh; offv = b * svb + h * svh
    q = tl.load(q_ptr + offq + rm[:, None] * sqt + rd[None, :], mask=rm[:, None] < T, other=0.0).to(tl.float16)
    m_i = tl.full((BM,), -float("inf"), tl.float32); l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    hi = tl.minimum((pm + 1) * BM, T)
    for n0 in range(0, hi, BN):
        cn = n0 + rn
        kt = tl.load(k_ptr + offk + cn[None, :] * skt + rd[:, None], mask=cn[None, :] < T, other=0.0).to(tl.float16)
        s = tl.dot(q, kt) * qk_scale
        if n0 + BN > pm * BM:   # (the diagonal tiles: causal mask)
            s = tl.where(rm[:, None] >= cn[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        al = tl.exp2(m_i - m_new)
        l_i = l_i * al + tl.sum(p, 1)
        v = tl.load(v_ptr + offv + cn[:, None] * svt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float16)
        acc = _dot16(p.to(tl.float16), v, acc * al[:, None], F16)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + b * sob + h * soh + rm[:, None] * sot + rd[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=rm[:, None] < T)
    tl.store(lse_ptr + bh * T + rm, m_i + tl.log2(l_i), mask=rm < T)


_PH = {}


def _philox(dev):
    """the (seed, offset) pair aten's flash backward takes (unused at dropout 0), from one tiny forward per device"""
    if dev not in _PH:
        z = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16, device=dev)
        r = torch.ops.aten._scaled_dot_product_flash_attention(z, z, z, 0.0, True, False)
        _PH[dev] = (r[6], r[7])
    return _PH[dev]


def _gl(x):
    """an output [B, H, T, D] laid out as [B, T, H, D] (the model's layout; the gradients share it: strides G)"""
    B, H, T, D = x.shape
    return torch.empty(B, T, H, D, dtype=x.dtype, device=x.device).transpose(1, 2)


def _st(q, k, v, g):
    return tuple(t.stride(i) for t in (q, k, v, g) for i in (0, 1, 2))


def _bhtd(x):
    """[B, H, T, D] with unit stride along D and a common (b, h, t) stride layout for q, k, v, o"""
    return x if x.stride(-1) == 1 else x.contiguous()


class F16AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, H, T, D = q.shape
        q, k, v = _bhtd(q), _bhtd(k), _bhtd(v)
        sm = D ** -0.5
        o = _gl(q)
        ST = _st(q, k, v, o)
        lse = torch.empty(B * H, T, dtype=torch.float32, device=q.device)
        BM, BN, nw, ns = TILES["fwd"]
        _fwd_kernel[(triton.cdiv(T, BM), B * H)](q, k, v, o, lse, sm * LOG2E, T, H, *ST, o.stride(0), o.stride(1), o.stride(2), D=D, BM=BM,
                                                 BN=BN, F16=F16ACC[0], num_warps=nw, num_stages=ns)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        ps, po = _philox(q.device)   # (lse: base 2 -> natural, flash's [B, H, T] layout)
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            do if do.stride(-1) == 1 else do.contiguous(), q, k, v, o, (lse * 0.6931471805599453).view(B, H, T),
            None, None, T, T, 0.0, True, ps, po)
        return dq, dk, dv


def attention(q, k, v):
    """causal softmax(q k^T / sqrt(D)) v, q k v [B, H, T, D] bf16 (one shared layout)"""
    return F16AttnFn.apply(q, k, v)
