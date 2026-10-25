"""Ternary BitNet b1.58 transformer (Llama-style) with master-free BitLinear.

Only the BitLinear matrices are ternary/master-free. The token embedding, the
RMSNorm gains, and the (tied) output projection stay full precision — together
~10% of params — and are trained by an ordinary tiny optimizer in the trainer.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .bitlinear import BitLinear


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


# elementwise pieces, swappable (tests)
_FN = {"rmsnorm": _rmsnorm, "swiglu": _swiglu, "rope": _rope_rot}


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
        self.wq = make_linear(c.dim, self.n_heads * self.head_dim)
        self.wk = make_linear(c.dim, self.n_kv * self.head_dim)
        self.wv = make_linear(c.dim, self.n_kv * self.head_dim)
        self.wo = make_linear(self.n_heads * self.head_dim, c.dim)

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, c: ModelConfig, make_linear):
        super().__init__()
        self.w_gate = make_linear(c.dim, c.hidden_dim)
        self.w_up = make_linear(c.dim, c.hidden_dim)
        self.w_down = make_linear(c.hidden_dim, c.dim)

    def forward(self, x):
        return self.w_down(_FN["swiglu"](self.w_gate(x), self.w_up(x)))


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

    def _block(self, x, freqs_cis):
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
        elif ckpt:
            x = x + checkpoint(self._attn, x, freqs_cis, use_reentrant=False)
            x = x + checkpoint(self._ffn, x, use_reentrant=False)
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
        if make_linear is None:
            make_linear = lambda i, o: BitLinear(i, o, c.act_bits)
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
        """Params NOT handled by the master-free BitLinear hooks (small tail).

        BitLinear stores its weight as int8 *buffers*, so it contributes no
        nn.Parameter. Everything in .parameters() is therefore the float tail:
        token embedding, RMSNorm gains, and (if untied) the lm_head.
        """
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
