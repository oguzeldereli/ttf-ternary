"""--head-fused: the loss of a FROZEN head (conversion recovery: the embedding / head do not train) computed with its
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
    assert not w.requires_grad, "--head-fused: the head must be frozen (its gradient is not computed)"
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
    assert not w.requires_grad, "--head-fused: the head must be frozen (its gradient is not computed)"
    return _HeadLossTopkFn.apply(h, w, tgt, ids, tlp, float(a), mode)


class _HeadLossTrainFn(torch.autograd.Function):
    """as _HeadLossFn for a TRAINED head: the forward also adds this chunk's dL/dW = dz^T h into an fp32 accumulator
    shared by all chunks of one loss (acc["dW"], vocab x d), and the first chunk backwarded returns acc * grad_out for W
    (the others None). Exact only because every chunk's grad_out is the same scalar (the chunks are summed, then
    divided by the token count): checked at each backward."""
    @staticmethod
    def forward(ctx, h, w, tgt, tlp, a, acc):
        with torch.no_grad():
            loss, dz = ce_kd_dz(F.linear(h, w).float(), tgt, tlp, a)
            dzb = dz.to(h.dtype)
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
                assert torch.allclose(acc["go"].float(), go.float(), rtol=1e-6), "--head-fused: unequal chunk scales"
            gw = None
        acc["left"] = acc.get("left", acc["n"]) - 1
        if acc["left"] == 0:
            acc.clear()                        # the accumulator freed after the last chunk's backward
        return (dh * go.float()).to(ctx.h_dtype), gw, None, None, None, None


def head_loss_train_chunk(h, w, tgt, tlp, a, acc):
    return _HeadLossTrainFn.apply(h, w, tgt, tlp, float(a), acc)
