"""The from-scratch transformer (Llama-style). Its linear layers come from make_linear: TTF's KernelTernaryLinear
(layers.build_kernel_transformer), the BitNet baseline's MasterTernaryLinear (master.build_master_transformer) or
nn.Linear (--mode fp32). The token embedding, the RMSNorm gains and the (tied) output projection stay full precision
and are trained by the trainer's float optimizer.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


def _rmsnorm(x, w, eps: float):
    # compute in fp32, return in the INPUT dtype: returning the weight dtype
    # promoted the whole residual stream to fp32 whenever the norm gain is fp32.
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * w.float()).to(dt)


def _swiglu(a, b):
    return F.silu(a) * b


def _rope_rot(x, freqs_cis):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f = freqs_cis[: x.shape[1]].unsqueeze(0).unsqueeze(2)  # [1,T,1,dim/2]
    return torch.view_as_real(xc * f).flatten(-2).type_as(x)


FUSED_ELEM = [True]   # bf16 CUDA activations: the RMSNorm and the SwiGLU in one Triton kernel each way (kernel.rms_norm,
                      # kernel.swiglu: the same fp32 formulas and bf16 roundings as the torch pieces above, which stay
                      # the CPU / fp32 path and the tests' reference)


ATTN_FP8 = [False]    # --attn-fp8 (fp8attn.py)
FUSE_PROJ = [""]      # --fuse-proj split | shared: q/k/v and gate/up each one ternary layer ("" off)
HEAD_ALL = [False]    # --head-fused: headloss.TrainHeadAll (the head's backward in the forward; fp32 weight-gradient sum)
ATTN_F16 = [False]    # --attn-f16 (f16attn.py)
FUSE_SWI = [False]    # --fuse-swiglu: silu(gate) up formed in w_down's prologue, its backward in w_down's dx epilogue
FUSE_RES = [False]    # --fuse-res: the residual add in the sublayer's last ternary GEMM, its gradient in the norm's backward
FUSED_HEAD = [True]   # training, no teacher, bf16 CUDA: headloss.TrainHeadCE per loss chunk (else the checkpointed autograd chunk)


def _fused(x):
    return FUSED_ELEM[0] and x.is_cuda and x.dtype == torch.bfloat16


def _rmsnorm_k(x, w, eps: float):
    if _fused(x):
        from .kernel import rms_norm
        return rms_norm(x, w, eps)
    return _rmsnorm(x, w, eps)


def _swiglu_k(a, b):
    if _fused(a) and b.dtype == a.dtype and a.shape == b.shape:
        from .kernel import swiglu
        return swiglu(a, b)
    return _swiglu(a, b)


def _rope_k(x, freqs_cis):
    if _fused(x) and x.dim() == 4 and (x.shape[-1] // 2) & (x.shape[-1] // 2 - 1) == 0:   # (head dim / 2 a power of 2)
        from .kernel import rope
        return rope(x, freqs_cis)
    return _rope_rot(x, freqs_cis)


# elementwise pieces, swappable (tests)
_FN = {"rmsnorm": _rmsnorm_k, "swiglu": _swiglu_k, "rope": _rope_k}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return _FN["rmsnorm"](x, self.weight, self.eps)


def precompute_rope(dim: int, end: int, theta: float):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex [end, dim/2]


def apply_rope(xq, xk, freqs_cis):
    return _FN["rope"](xq, freqs_cis), _FN["rope"](xk, freqs_cis)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig, make_linear):
        super().__init__()
        self.n_heads = c.n_heads
        self.n_kv = c.n_kv_heads
        self.head_dim = c.dim // c.n_heads
        self.rep = self.n_heads // self.n_kv
        if FUSE_PROJ[0]:   # --fuse-proj: q, k, v as one [Nq + 2 Nkv, dim] layer (one prologue, GEMM, dx; shared column scale)
            self.wqkv = make_linear(c.dim, (self.n_heads + 2 * self.n_kv) * self.head_dim)
            self.wqkv.row_blocks = [self.n_heads * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim]
        else:
            self.wq = make_linear(c.dim, self.n_heads * self.head_dim)
            self.wk = make_linear(c.dim, self.n_kv * self.head_dim)
            self.wv = make_linear(c.dim, self.n_kv * self.head_dim)
        self.wo = make_linear(self.n_heads * self.head_dim, c.dim)

    def forward(self, x, freqs_cis, res=None):
        B, T, _ = x.shape
        if hasattr(self, "wqkv"):
            q, k, v = self.wqkv(x).split(self.wqkv.row_blocks, dim=-1)
            q = q.reshape(B, T, self.n_heads, self.head_dim)
            k = k.reshape(B, T, self.n_kv, self.head_dim)
            v = v.reshape(B, T, self.n_kv, self.head_dim)
        else:
            q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
            k = self.wk(x).view(B, T, self.n_kv, self.head_dim)
            v = self.wv(x).view(B, T, self.n_kv, self.head_dim)
        if getattr(self, "qk_logscale", None) is not None:
            # learnable per-head temperature (--qk-temp): logits scale by exp(s_h)
            q = q * self.qk_logscale.exp().view(1, 1, -1, 1).to(q.dtype)
        q, k = apply_rope(q, k, freqs_cis)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=2)
            v = v.repeat_interleave(self.rep, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,T,hd]
        if ATTN_F16[0] and q.is_cuda and q.dtype == torch.bfloat16 and torch.is_grad_enabled():
            from .f16attn import attention as _fa16     # --attn-f16: own flash fwd+bwd, fp16-accumulated softmax side
            out = _fa16(q, k, v)
        elif ATTN_FP8[0] and q.is_cuda and q.dtype == torch.bfloat16 and torch.is_grad_enabled():
            from .fp8attn import attention as _fa8      # --attn-fp8: the forward on FP8 tensor cores
            out = _fa8(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out) if res is None else self.wo(out, acc=res)


class FeedForward(nn.Module):
    def __init__(self, c: ModelConfig, make_linear):
        super().__init__()
        if FUSE_PROJ[0]:   # --fuse-proj: gate and up as one [2 hidden, dim] layer
            self.w_gu = make_linear(c.dim, 2 * c.hidden_dim)
            self.w_gu.row_blocks = [c.hidden_dim, c.hidden_dim]
        else:
            self.w_gate = make_linear(c.dim, c.hidden_dim)
            self.w_up = make_linear(c.dim, c.hidden_dim)
        self.w_down = make_linear(c.hidden_dim, c.dim)

    def forward(self, x, res=None):
        kw = {} if res is None else {"acc": res}
        if FUSE_SWI[0] and hasattr(self.w_down, "swi_ok") and self.w_down.swi_ok(x):
            if hasattr(self, "w_gu"):
                return self.w_down.forward_swiglu(self.w_gu(x), None, **kw)
            return self.w_down.forward_swiglu(self.w_gate(x), self.w_up(x), **kw)
        if hasattr(self, "w_gu"):
            g_, u_ = self.w_gu(x).chunk(2, dim=-1)
            return self.w_down(_FN["swiglu"](g_.contiguous(), u_.contiguous()), **kw)
        return self.w_down(_FN["swiglu"](self.w_gate(x), self.w_up(x)), **kw)


class MoEFeedForward(nn.Module):
    """--moe E:K[:H]: a mixture of E SwiGLU experts (hidden H each), top-K routing by a float router. Each expert's three
    linears are ternary layers of their own, so each gets its own TTF state, trained only by the tokens routed to it
    (its gradient comes from those tokens alone). Switch-style load-balancing loss E * sum_e f_e P_e, kept in aux_loss for
    the model to add (weight BitTransformer.moe_aux_w)."""

    def __init__(self, dim, n_exp, top_k, hidden, make_linear):
        super().__init__()
        self.E, self.k = n_exp, top_k
        self.router = nn.Linear(dim, n_exp, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = nn.ModuleList()
        for _ in range(n_exp):
            e = nn.Module()
            e.w_gate = make_linear(dim, hidden); e.w_up = make_linear(dim, hidden); e.w_down = make_linear(hidden, dim)
            self.experts.append(e)
        self.aux_loss = None
        self.load = None

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        logits = self.router(xf).float()
        probs = logits.softmax(-1)
        w, idx = probs.topk(self.k, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(xf)
        frac = torch.zeros(self.E, device=x.device)
        for e, ex in enumerate(self.experts):
            sel = (idx == e)
            rows = sel.any(-1).nonzero().squeeze(1)
            if rows.numel() == 0:
                continue
            frac[e] = rows.numel() / xf.shape[0]
            we = (w * sel).sum(-1)[rows]
            xe = xf[rows]
            ye = ex.w_down(_FN["swiglu"](ex.w_gate(xe), ex.w_up(xe)))
            out.index_add_(0, rows, (ye * we[:, None]).to(out.dtype))   # experts may return fp32 (row scales)
        self.aux_loss = self.E * (frac.detach() * probs.mean(0)).sum() / self.k
        self.load = frac.detach()
        return out.view(B, T, d)


def enable_moe(model, spec, make_linear):
    """replace every block's feed-forward by an MoEFeedForward (spec E:K[:H], H defaults to the dense hidden size)"""
    parts = [int(x) for x in spec.split(":")]
    E, k = parts[0], parts[1]
    c = model.c
    H = parts[2] if len(parts) > 2 else c.hidden_dim
    dev = next(model.parameters()).device
    for blk in model.layers:
        blk.ffn = MoEFeedForward(c.dim, E, k, H, make_linear).to(dev)
    return E, k, H


class Block(nn.Module):
    def __init__(self, c: ModelConfig, make_linear):
        super().__init__()
        self.attn_norm = RMSNorm(c.dim, c.norm_eps)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        self.attn = Attention(c, make_linear)
        self.ffn = FeedForward(c, make_linear)

    def _attn(self, x, freqs_cis):
        return self.attn(self.attn_norm(x), freqs_cis)

    def _ffn(self, x):
        return self.ffn(self.ffn_norm(x))

    def _fres(self, x):
        """--fuse-res applies: fused elementwise kernels, ternary last layers that take the residual (acc), dense FFN"""
        return (FUSE_RES[0] and _fused(x) and getattr(self.attn.wo, "takes_acc", False)
                and isinstance(self.ffn, FeedForward) and getattr(self.ffn.w_down, "takes_acc", False)
                and getattr(self.attn_norm, "eps", None) is not None)

    def _attn_r(self, x, freqs_cis):   # x + attn(norm(x)), the add in wo's epilogue
        from .kernel import res_norm
        xr, n = res_norm(x, self.attn_norm.weight, self.attn_norm.eps)
        return self.attn(n, freqs_cis, res=xr)

    def _ffn_r(self, x):
        from .kernel import res_norm
        xr, n = res_norm(x, self.ffn_norm.weight, self.ffn_norm.eps)
        return self.ffn(n, res=xr)

    def _block(self, x, freqs_cis):
        if self._fres(x):
            return self._ffn_r(self._attn_r(x, freqs_cis))
        x = x + self._attn(x, freqs_cis)
        return x + self._ffn(x)

    def forward(self, x, freqs_cis, ckpt=False):
        # checkpoint attn and ffn separately: only one sublayer's dense weight +
        # gradient is materialized at a time during backward (much lower peak).
        # The norm goes INSIDE the checkpoint: it upcasts to fp32, and leaving it
        # outside retained two fp32 [B,T,d] tensors per sublayer (4.7 GiB at
        # 32k tokens/step, 12 layers) instead of recomputing them for free.
        if ckpt and getattr(self, "ckpt_block", False):
            # --ckpt block: one checkpoint per block (its input only): half the saved activations of the per-sublayer
            # checkpoints, for a larger recompute (attention + MLP of one block at once); for the ternary kernels,
            # which never build a dense weight
            x = checkpoint(self._block, x, freqs_cis, use_reentrant=False)
        elif ckpt and self._fres(x):
            x = checkpoint(self._attn_r, x, freqs_cis, use_reentrant=False)
            x = checkpoint(self._ffn_r, x, use_reentrant=False)
        elif ckpt:
            x = x + checkpoint(self._attn, x, freqs_cis, use_reentrant=False)
            x = x + checkpoint(self._ffn, x, use_reentrant=False)
        elif self._fres(x):
            x = self._ffn_r(self._attn_r(x, freqs_cis))
        else:
            x = x + self._attn(x, freqs_cis)
            x = x + self._ffn(x)
        return x


class BitTransformer(nn.Module):
    # tokens per output-head chunk: the [tokens, vocab] logits are the largest
    # tensor in the model (32k vocab x 16k tokens = 2 GiB in fp32), so the head +
    # cross-entropy run in checkpointed chunks and full logits are never stored.
    loss_chunk = 2048
    # partial gradient checkpointing: the last `ckpt_skip` layers keep their activations
    # (no forward recompute in backward) -- trades memory for ~1/3 of those layers' cost
    ckpt_skip = 0

    def __init__(self, c: ModelConfig, grad_checkpoint: bool = True, make_linear=None):
        super().__init__()
        self.c = c
        self.grad_checkpoint = grad_checkpoint
        assert make_linear is not None, "BitTransformer: make_linear builds the linear layers (see the module docstring)"
        self.tok_emb = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleList([Block(c, make_linear) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.dim, c.norm_eps)
        if c.tie_embeddings:
            self.lm_head = None  # logits = h @ tok_emb.weight.T
        else:
            self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.register_buffer(
            "freqs_cis",
            precompute_rope(c.dim // c.n_heads, c.max_seq_len, c.rope_theta),
            persistent=False,
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _loss_chunk(h, w, tgt):
        return F.cross_entropy(F.linear(h, w).float(), tgt, ignore_index=-1,
                               reduction="sum")

    # --distill: the teacher's log-probabilities for this batch ([tokens, vocab]), set by the trainer around one
    # forward / backward; the loss becomes (1 - kd_w) CE + kd_w KL(teacher || student), per token
    kd_logp = None
    kd_w = 1.0
    moe_aux_w = 0.01      # --moe: weight of the load-balancing loss

    @staticmethod
    def _loss_chunk_kd(h, w, tgt, tlp, a):
        z = F.linear(h, w).float()
        lp = F.log_softmax(z, -1)
        kl = F.kl_div(lp, tlp.float(), log_target=True, reduction="sum")
        if a >= 1.0:
            return kl
        return (1 - a) * F.nll_loss(lp, tgt, ignore_index=-1, reduction="sum") + a * kl

    def forward(self, idx, targets=None):
        h = self.tok_emb(idx)
        if torch.is_autocast_enabled(h.device.type):
            # keep the residual stream in the autocast dtype even when the
            # embedding is fp32 (master mode), so activation memory and the math
            # match across modes.
            h = h.to(torch.get_autocast_dtype(h.device.type))
        fc = self.freqs_cis.to(h.device)
        ckpt = self.grad_checkpoint and self.training
        n_ck = len(self.layers) - self.ckpt_skip
        for i, layer in enumerate(self.layers):
            h = layer(h, fc, ckpt=ckpt and i < n_ck)
        h = self.norm(h)
        from .intk_emb import factored_head, head_loss as _ik_loss
        ik = factored_head(self.tok_emb if self.lm_head is None else self.lm_head, self.training)
        if ik is not None and targets is not None:
            # --intk-emb, factored update: the int-k head's chunked loss (no [V, d] matrix, no dense dL/dW)
            hf, tf = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
            kd = ("lp", self.kd_logp, self.kd_w) if self.kd_logp is not None else None
            loss = _ik_loss(ik, hf, tf, kd, self.loss_chunk) / (tf != -1).sum().clamp_min(1)
            aux = [l.ffn.aux_loss for l in self.layers if getattr(l.ffn, "aux_loss", None) is not None]
            if aux and self.moe_aux_w:
                loss = loss + self.moe_aux_w * torch.stack(aux).mean()
            return None, loss
        w = self.tok_emb.weight if self.lm_head is None else self.lm_head.weight
        if targets is None:
            return F.linear(h, w), None
        # chunked head + loss; sum/count, so the result is a per-token mean that
        # ignores padding (ignore_index=-1) exactly. Returns logits=None.
        hf = h.reshape(-1, h.shape[-1])
        tf = targets.reshape(-1)
        n = self.loss_chunk
        total = hf.new_zeros((), dtype=torch.float32)
        kd = self.kd_logp
        if (kd is None and self.training and FUSED_HEAD[0] and HEAD_ALL[0] and hf.is_cuda and hf.dtype == torch.bfloat16
                and torch.is_grad_enabled()):
            from .headloss import TrainHeadAll     # (--head-fused: the head's backward GEMMs in the forward, no dz kept)
            total = TrainHeadAll.apply(hf, w, tf, n)
        elif kd is None and self.training and FUSED_HEAD[0] and hf.is_cuda and hf.dtype == torch.bfloat16:
            from .headloss import TrainHeadCE      # (the head GEMM once per chunk, the softmax's gradient formed with the loss)
            wb = w.to(hf.dtype)
            for i in range(0, hf.shape[0], n):
                total = total + TrainHeadCE.apply(hf[i:i + n], wb, tf[i:i + n])
        else:
            for i in range(0, hf.shape[0], n):
                hc, tc = hf[i:i + n], tf[i:i + n]
                if kd is not None:
                    tl = kd[i:i + n]
                    total = total + (checkpoint(self._loss_chunk_kd, hc, w, tc, tl, self.kd_w, use_reentrant=False)
                                     if ckpt else self._loss_chunk_kd(hc, w, tc, tl, self.kd_w))
                    continue
                total = total + (checkpoint(self._loss_chunk, hc, w, tc, use_reentrant=False)
                                 if ckpt else self._loss_chunk(hc, w, tc))
        loss = total / (tf != -1).sum().clamp_min(1)
        aux = [l.ffn.aux_loss for l in self.layers if getattr(l.ffn, "aux_loss", None) is not None]
        if aux and self.moe_aux_w:
            loss = loss + self.moe_aux_w * torch.stack(aux).mean()
        return None, loss

    def float_tail_parameters(self):
        """The trainable parameters (requires_grad): with TTF layers (trits in buffers) the float tail: token
        embedding, RMSNorm gains and (if untied) the lm_head."""
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.c.max_seq_len:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
