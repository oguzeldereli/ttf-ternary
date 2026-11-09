"""The fused head loss (QwenTern.head_fused, the default): the loss of a FROZEN head (conversion recovery: the embedding / head do not train) computed with its
gradient in one pass. A checkpointed chunk computes z = h W^T and the loss in the forward, then again in the backward
(the recompute) before the softmax backward and dh = dz W: two 248k-wide GEMMs + softmaxes per chunk in the forward
direction. Here the forward computes the loss AND dL/dh = dz W at once and saves only dL/dh (chunk x d), so the
backward is a scale: one 248k-wide GEMM pair per chunk instead of three, no saved logits.
  dz (per token) = softmax(z) * ((1 - a) [t valid] + a sum(p_t)) - (1 - a) onehot(t) - a p_t
for the loss (1 - a) CE + a KL(p_t || softmax(z)) (p_t the teacher's distribution, its log-probabilities given), the
same function as BitTransformer._loss_chunk_kd; a = 0 (no teacher): cross-entropy, as qwen._ce_chunk. Same maths, a
different order of operations: not bit-identical to the autograd path (equivalence: tests/checks)."""
import os
import torch
import torch.nn.functional as F

_CHECK = os.environ.get("TERN_CHECK_HEAD") == "1"


def ce_kd_dz(z, tgt, tlp, a):
    """the summed loss of one chunk's logits z [n, V] (fp32) and dL/dz: (1 - a) CE + a KL(p_t || softmax(z)) (tlp the
    teacher's log-probabilities), or CE (tlp None); z is consumed (overwritten)"""
    lp = F.log_softmax(z, -1)
    del z
    ok = tgt != -1
    nll = -lp.gather(1, tgt.clamp_min(0)[:, None])[:, 0][ok].sum()
    p = lp.exp()
    if tlp is not None:
        tl = tlp.float(); pt = tl.exp()
        kl = (pt * (tl - lp)).sum(); del tl
        loss = kl if a >= 1.0 else (1 - a) * nll + a * kl
        ca = 0.0 if a >= 1.0 else (1 - a)
        ak = min(a, 1.0)                  # a >= 1: the loss is the KL alone, weight 1 (as _loss_chunk_kd)
        dz = p * (ca * ok.float()[:, None] + ak * pt.sum(-1, keepdim=True)) - ak * pt
    else:
        loss, ca = nll, 1.0
        dz = p * ok.float()[:, None]
    if ca:
        idx = tgt.clamp_min(0)[:, None]
        dz.scatter_add_(1, idx, torch.full_like(idx, -ca, dtype=dz.dtype) * ok[:, None])
    return loss, dz


try:
    import triton
    import triton.language as tl
except ImportError:          # (CPU-only installs: the torch path below)
    triton = None

FUSED_CEKD = [os.environ.get("TTF_HEAD_TORCH") != "1"]   # the Triton kernel for ce_kd_dz on CUDA bf16 logits (9 Nov)

if triton is not None:
    @triton.jit
    def _cekd_kernel(z_ptr, t_ptr, tgt_ptr, dz_ptr, nll_ptr, kl_ptr, V, sz, st, sd, ca, ak,
                     HAS_T: tl.constexpr, BLOCK: tl.constexpr):
        """one row: lse of z (two passes: max, then sum), the teacher's sums, then dz in a third pass; z / tl read
        as bf16 and computed in fp32 (as the torch path's .float()), dz written as bf16 (its only use)"""
        row = tl.program_id(0).to(tl.int64)
        zr = z_ptr + row * sz
        tr = t_ptr + row * st
        dr = dz_ptr + row * sd
        tgt = tl.load(tgt_ptr + row)
        ok = tgt != -1
        tg = tl.where(ok, tgt, 0)
        offs = tl.arange(0, BLOCK)
        mx = tl.full([BLOCK], float("-inf"), tl.float32)
        for v0 in range(0, V, BLOCK):
            o = v0 + offs
            mx = tl.maximum(mx, tl.load(zr + o, mask=o < V, other=float("-inf")).to(tl.float32))
        m = tl.max(mx, 0)
        se = tl.zeros([BLOCK], tl.float32)
        spt = tl.zeros([BLOCK], tl.float32)
        sptd = tl.zeros([BLOCK], tl.float32)
        for v0 in range(0, V, BLOCK):
            o = v0 + offs
            msk = o < V
            zb = tl.load(zr + o, mask=msk, other=float("-inf")).to(tl.float32)
            se += tl.exp(zb - m)
            if HAS_T:
                tb = tl.load(tr + o, mask=msk, other=float("-inf")).to(tl.float32)
                pt = tl.exp(tb)
                spt += pt
                sptd += tl.where(msk, pt * (tb - zb), 0.0)     # sum pt (tl - z): KL = this + lse sum pt
        lse = m + tl.log(tl.sum(se, 0))
        zt = tl.load(zr + tg).to(tl.float32)
        tl.store(nll_ptr + row, tl.where(ok, lse - zt, 0.0))
        S = 0.0
        if HAS_T:
            S = tl.sum(spt, 0)
            tl.store(kl_ptr + row, tl.sum(sptd, 0) + lse * S)
        okf = tl.where(ok, 1.0, 0.0)
        coef = ca * okf + ak * S
        for v0 in range(0, V, BLOCK):
            o = v0 + offs
            msk = o < V
            zb = tl.load(zr + o, mask=msk, other=float("-inf")).to(tl.float32)
            d = tl.exp(zb - lse) * coef
            if HAS_T:
                tb = tl.load(tr + o, mask=msk, other=float("-inf")).to(tl.float32)
                d -= ak * tl.exp(tb)
            d -= tl.where(o == tg, ca * okf, 0.0)
            tl.store(dr + o, d.to(tl.bfloat16), mask=msk)


def ce_kd_dz_fused(zb, tgt, tlp, a):
    """ce_kd_dz from bf16 logits zb [n, V] on CUDA (the Triton kernel): (loss, dz bf16). Same maths; the sums in
    another order (fp32), dz rounded to bf16 once (the torch path rounds its fp32 dz to bf16 for dh)"""
    n, V = zb.shape
    zb = zb.contiguous()
    t_ = tlp.contiguous() if tlp is not None else zb
    dz = torch.empty_like(zb, dtype=torch.bfloat16)
    nll = torch.empty(n, device=zb.device, dtype=torch.float32)
    kl = torch.empty(n, device=zb.device, dtype=torch.float32)
    if tlp is not None:
        ca, ak = (0.0 if a >= 1.0 else 1 - a), min(a, 1.0)
    else:
        ca, ak = 1.0, 0.0
    _cekd_kernel[(n,)](zb, t_, tgt.contiguous(), dz, nll, kl, V, zb.stride(0), t_.stride(0), dz.stride(0),
                       float(ca), float(ak), HAS_T=tlp is not None, BLOCK=4096, num_warps=8)
    nl = nll.sum()
    if tlp is None:
        return nl, dz
    k = kl.sum()
    return (k if a >= 1.0 else (1 - a) * nl + a * k), dz


def _fused_ok(h, w):
    return FUSED_CEKD[0] and triton is not None and h.is_cuda and h.dtype == torch.bfloat16 and w.dtype == torch.bfloat16


def topk_dz(z, tgt, ids, tlp, a, mode):
    """as ce_kd_dz for qwen._kd_topk_chunk ((1 - a) CE + a KD to the teacher's top k, mode 'rest' / 'renorm')"""
    lse = torch.logsumexp(z, -1, keepdim=True)
    s = (z - lse).exp()                                           # the student's softmax
    lq = z.gather(1, ids) - lse
    if mode == "renorm":
        q = torch.softmax(tlp.float(), -1)
        kd = -(q * lq).sum()
        dz = s * q.sum(-1, keepdim=True)
        dz.scatter_add_(1, ids, -q)
    else:
        pk = tlp.float().exp(); p_rest = (1 - pk.sum(-1, keepdim=True)).clamp_min(0)
        zm = z.scatter(1, ids, float("-inf")); lse_r = torch.logsumexp(zm, -1, keepdim=True)
        kd = -(pk * lq).sum() - (p_rest * (lse_r - lse)).sum()
        dz = s * (pk.sum(-1, keepdim=True) + p_rest) - p_rest * (zm - lse_r).exp()
        dz.scatter_add_(1, ids, -pk)
        del zm
    if a >= 1.0:
        return kd, dz
    ok = tgt != -1
    ce = -(z.gather(1, tgt.clamp_min(0)[:, None]) - lse)[:, 0][ok].sum()
    dz.mul_(a).add_(s * ((1 - a) * ok.float()[:, None]))
    idx = tgt.clamp_min(0)[:, None]
    dz.scatter_add_(1, idx, torch.full_like(idx, -(1 - a), dtype=dz.dtype) * ok[:, None])
    return (1 - a) * ce + a * kd, dz


class _HeadLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, tgt, tlp, a):
        with torch.no_grad():
            if _fused_ok(h, w):     # bf16 logits straight into the kernel: no fp32 [n, V] tensors
                loss, dz = ce_kd_dz_fused(F.linear(h, w), tgt, tlp, a)
            else:
                loss, dz = ce_kd_dz(F.linear(h, w).float(), tgt, tlp, a)
            dh = torch.mm(dz.to(h.dtype), w, out_dtype=torch.float32) if h.dtype == torch.bfloat16 else dz.to(h.dtype) @ w
        ctx.save_for_backward(dh); ctx.h_dtype = h.dtype
        return loss

    @staticmethod
    def backward(ctx, go):
        (dh,) = ctx.saved_tensors
        return (dh * go.float()).to(ctx.h_dtype), None, None, None, None   # dL/dh kept fp32: one rounding, as autograd's


def head_loss_chunk(h, w, tgt, tlp=None, a=0.0):
    """the summed loss of one chunk (frozen head w): as _ce_chunk (tlp None) / _loss_chunk_kd (tlp the teacher's
    log-probabilities, weight a)"""
    assert not w.requires_grad, "head_loss: the head must be frozen (its gradient is not computed)"
    return _HeadLossFn.apply(h, w, tgt, tlp, float(a))


class _HeadLossTopkFn(torch.autograd.Function):
    """as qwen._kd_topk_chunk ((1 - a) CE + a KD to the teacher's top k; mode 'rest': the k tokens plus one bucket for
    the rest of the mass, 'renorm': the k renormalised), with dL/dh computed in the forward (frozen head)"""
    @staticmethod
    def forward(ctx, h, w, tgt, ids, tlp, a, mode):
        with torch.no_grad():
            loss, dz = topk_dz(F.linear(h, w).float(), tgt, ids, tlp, a, mode)
            dh = torch.mm(dz.to(h.dtype), w, out_dtype=torch.float32) if h.dtype == torch.bfloat16 else dz.to(h.dtype) @ w
        ctx.save_for_backward(dh); ctx.h_dtype = h.dtype
        return loss

    @staticmethod
    def backward(ctx, go):
        (dh,) = ctx.saved_tensors
        return (dh * go.float()).to(ctx.h_dtype), None, None, None, None, None, None


def head_loss_topk_chunk(h, w, tgt, ids, tlp, a, mode="rest"):
    assert not w.requires_grad, "head_loss: the head must be frozen (its gradient is not computed)"
    return _HeadLossTopkFn.apply(h, w, tgt, ids, tlp, float(a), mode)


class _HeadLossTrainFn(torch.autograd.Function):
    """as _HeadLossFn for a TRAINED head: the forward also adds this chunk's dL/dW = dz^T h into an fp32 accumulator
    shared by all chunks of one loss (acc["dW"], vocab x d), and the first chunk backwarded returns acc * grad_out for W
    (the others None). Exact only because every chunk's grad_out is the same scalar (the chunks are summed, then
    divided by the token count): checked at each backward."""
    @staticmethod
    def forward(ctx, h, w, tgt, tlp, a, acc):
        with torch.no_grad():
            if _fused_ok(h, w):
                loss, dzb = ce_kd_dz_fused(F.linear(h, w), tgt, tlp, a)
            else:
                loss, dz = ce_kd_dz(F.linear(h, w).float(), tgt, tlp, a)
                dzb = dz.to(h.dtype); del dz
            dh = torch.mm(dzb, w, out_dtype=torch.float32) if h.dtype == torch.bfloat16 else dzb @ w
            dW = torch.mm(dzb.T, h, out_dtype=torch.float32) if h.dtype == torch.bfloat16 else dzb.T.float() @ h.float()
            if acc.get("dW") is None:
                acc["dW"] = dW
            else:
                acc["dW"].add_(dW)
            acc["n"] = acc.get("n", 0) + 1
        ctx.save_for_backward(dh); ctx.h_dtype, ctx.w_dtype, ctx.acc = h.dtype, w.dtype, acc
        return loss

    @staticmethod
    def backward(ctx, go):
        (dh,) = ctx.saved_tensors
        acc = ctx.acc
        if "go" not in acc:
            acc["go"] = go.detach(); gw = (acc["dW"] * go.float()).to(ctx.w_dtype)
        else:
            if _CHECK:                         # (a host sync: only when asked)
                assert torch.allclose(acc["go"].float(), go.float(), rtol=1e-6), "head_loss_train_chunk: unequal chunk scales"
            gw = None
        acc["left"] = acc.get("left", acc["n"]) - 1
        if acc["left"] == 0:
            acc.clear()                        # the accumulator freed after the last chunk's backward
        return (dh * go.float()).to(ctx.h_dtype), gw, None, None, None, None


def head_loss_train_chunk(h, w, tgt, tlp, a, acc):
    return _HeadLossTrainFn.apply(h, w, tgt, tlp, float(a), acc)
