"""--head-fused: the loss of a FROZEN head (conversion recovery: the embedding / head do not train) computed with its
gradient in one pass. A checkpointed chunk computes z = h W^T and the loss in the forward, then again in the backward
(the recompute) before the softmax backward and dh = dz W: two 248k-wide GEMMs + softmaxes per chunk in the forward
direction. Here the forward computes the loss AND dL/dh = dz W at once and saves only dL/dh (chunk x d), so the
backward is a scale: one 248k-wide GEMM pair per chunk instead of three, no saved logits.
  dz (per token) = softmax(z) * ((1 - a) [t valid] + a sum(p_t)) - (1 - a) onehot(t) - a p_t
for the loss (1 - a) CE + a KL(p_t || softmax(z)) (p_t the teacher's distribution, its log-probabilities given), the
same function as BitTransformer._loss_chunk_kd; a = 0 (no teacher): cross-entropy, as qwen._ce_chunk. Same maths, a
different order of operations: not bit-identical to the autograd path (equivalence: tests/checks)."""
import torch
import torch.nn.functional as F


class _HeadLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, tgt, tlp, a):
        with torch.no_grad():
            z = F.linear(h, w).float()
            lp = F.log_softmax(z, -1)
            ok = tgt != -1
            nll = -lp.gather(1, tgt.clamp_min(0)[:, None])[:, 0][ok].sum()
            p = lp.exp()
            if tlp is not None:
                pt = tlp.float().exp()
                kl = (pt * (tlp.float() - lp)).sum()
                loss = kl if a >= 1.0 else (1 - a) * nll + a * kl
                ca = 0.0 if a >= 1.0 else (1 - a)
                dz = p * (ca * ok.float()[:, None] + a * pt.sum(-1, keepdim=True)) - a * pt
            else:
                loss, ca = nll, 1.0
                dz = p * ok.float()[:, None]
            if ca:
                idx = tgt.clamp_min(0)[:, None]
                dz.scatter_add_(1, idx, torch.full_like(idx, -ca, dtype=dz.dtype) * ok[:, None])
            dh = dz.to(h.dtype) @ w
        ctx.save_for_backward(dh)
        return loss

    @staticmethod
    def backward(ctx, go):
        (dh,) = ctx.saved_tensors
        return (dh * go.to(dh.dtype)), None, None, None, None


def head_loss_chunk(h, w, tgt, tlp=None, a=0.0):
    """the summed loss of one chunk (frozen head w): as _ce_chunk (tlp None) / _loss_chunk_kd (tlp the teacher's
    log-probabilities, weight a)"""
    assert not w.requires_grad, "--head-fused: the head must be frozen (its gradient is not computed)"
    return _HeadLossFn.apply(h, w, tgt, tlp, float(a))
