"""Qwen3.8-Flash-Next text decoder (HF model_type qwen4_exp) for conversion and TTF training, 5 Nov 2026.

User, 5 Nov ~01:20 (via P, confirmed): "convert and train Qwen 3.8 Flash Next ... r16 adapters, row col scales, an
optimal theta and float lr, and train for 1k steps, rank 0.2-0.3 whatever fits. Use mix5c for now".

transformers 5.19 (the lab / box env) has Qwen3.5-MoE but not qwen4_exp, so this file builds the text decoder from
5.19's Qwen3.5-MoE parts (gated DeltaNet, gated attention, router, shared expert) plus the qwen4_exp-only pieces,
written here after the reference (transformers main, models/qwen4_exp/modular_qwen4_exp.py):
  gated residual: hc_count = 4 residual streams of hidden_size; each sub-block reads a sigmoid-gated mean of the grouped-
    RMS-normed streams (rank-320 gate) and writes its output into every stream with a per-stream scalar weight
  PLE: hashed 2/3-gram embedding (16 heads x 160, 51B rows x dims, FP8) at layer index 1, gated per stream by the
    normed streams, plus a dilated depthwise conv
  QSA: the full-attention layers' indexer selects 2048 tokens from blocks of 4; for T <= 2048 it selects every visible
    token, so the layer is dense causal attention (we assert T <= indexer_budget and skip it; its weights are kept)
  the DeltaNet output gate is sigmoid (output_gate_type), the 5.19 class's is silu
Dropped: vision tower, MTP layer.

Per-layer API (P's layer-major trainer, ttf-lab/wip/moe_ttf.py):
  m.embed(ids) -> h0 [B, T, hc x d];  m.rope(T, dev) -> (cos, sin);  m.layers[i](h, pe, ids) -> h;
  m.head_in(h) -> [B, T, d] (the stream mixer; there is no other final norm);  m.lm_head (float, untied)
The routed experts are two modules mlp.gu [E, 2I, d] (gate rows, then up) and mlp.dn [E, d, I], each forward(x_sorted
[T, K], offsets [E + 1]) -> [T, N]; DenseMat below is the float reference (and the conversion's source), P's
TernaryExperts the trained one. Router sort / unsort and the weighted sum live in MoE.forward.
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as Q
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

EXTRA = ("hc_count", "hc_lowrank", "ple_layer_ids", "ple_embed_dim", "ple_conv_kernel_size", "ngram_size",
         "heads_per_ngram", "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "split_ngram_parts",
         "indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio",
         "output_gate_type", "seed")
DEFAULTS = {"seed": 1234, "split_ngram_parts": 512, "make_ngram_vocab_size_divisible_by": 128, "ple_layer_ids": []}


def text_config(path_or_dict):
    """5.19's Qwen3.5-MoE text config filled from qwen4_exp's text_config, the extra fields as attributes"""
    if isinstance(path_or_dict, dict):
        c = path_or_dict
    else:
        c = json.load(open(os.path.join(path_or_dict, "config.json")))
    t = dict(c.get("text_config", c))
    t["layer_types"] = ["full_attention" if x in ("full_attention", "indexed_attention") else x for x in t["layer_types"]]
    known = {k: v for k, v in t.items() if k not in EXTRA and k not in ("mtp", "model_type", "dtype")}
    cfg = Qwen3_5MoeTextConfig(**known)
    for k in EXTRA:
        setattr(cfg, k, t.get(k, DEFAULTS.get(k)))
    if cfg.ple_embed_dim is None:
        cfg.ple_embed_dim = cfg.hidden_size
    cfg._attn_implementation = "sdpa"
    return cfg


class GroupRMSNorm(nn.Module):
    """zero-centred RMSNorm (x / rms * (1 + w)), the rms taken per group of `group` channels"""

    def __init__(self, dim, group=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group, self.eps = group, eps

    def forward(self, x):
        y = x.float()
        if self.group is not None:
            y = y.unflatten(-1, (-1, self.group))
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group is not None:
            y = y.flatten(-2)
        return (y * (1.0 + self.weight.float())).type_as(x)


class GatedResidual(nn.Module):
    def __init__(self, cfg, combine=True):
        super().__init__()
        self.hc, self.d = cfg.hc_count, cfg.hidden_size
        D = self.hc * self.d
        self.hc_norm = GroupRMSNorm(D, self.d, cfg.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(D, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, D, bias=False)
        self.block_inject_weight = nn.Linear(D, self.hc, bias=False) if combine else None

    def forward(self, h):
        hn = self.hc_norm(h)
        g = torch.sigmoid(self.input_mix_weight_up(F.silu(self.input_mix_weight_down(hn) / self.hc)))
        mixed = (g.unflatten(-1, (self.hc, self.d)) * hn.unflatten(-1, (self.hc, self.d))).mean(-2)
        if self.block_inject_weight is None:
            return mixed
        return mixed, 2 * torch.sigmoid(self.block_inject_weight(hn) / self.hc)

    @staticmethod
    def inject(h, y, w):
        """h + y written into every stream with weight w [.., hc]"""
        return h + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)


def compile_hc():
    """torch.compile the hyper-connection glue (GatedResidual's grouped RMS norm, gates, stream mix and inject; the
    MoE's SwiGLU): fused elementwise kernels with fp32 inside instead of bf16 intermediates (rounding level vs eager);
    row reductions only, no atomics (deterministic)"""
    GatedResidual.forward = torch.compile(GatedResidual.forward, dynamic=False)
    GatedResidual.inject = staticmethod(torch.compile(GatedResidual.inject, dynamic=False))
    global _swiglu
    _swiglu = torch.compile(_swiglu, dynamic=False)
    # the reference's eager norms in the mixers: DeltaNet's gated RMS norm (on [T heads, 128]) and attention's q / k
    # norms (on [.., heads, 256]): pow / mean / casts / silu as one kernel each way
    fused_conv()
    if COMPILE_NORMS[0]:
        Q.Qwen3_5MoeRMSNormGated.forward = torch.compile(Q.Qwen3_5MoeRMSNormGated.forward, dynamic=False)
        Q.Qwen3_5MoeRMSNorm.forward = torch.compile(Q.Qwen3_5MoeRMSNorm.forward, dynamic=False)


COMPILE_NORMS = [True]
FUSED_CONV = [True]     # CUDA bf16: DeltaNet's causal conv + silu as moe_gg.CConv on the [B, T, C] layout (no transposed
                        # copies; the reference's fallback is F.conv1d on a [B, C, T] copy, then silu)


def fused_conv():
    """route the reference's causal_conv1d_fn (the torch fallback here) through moe_gg.causal_conv1d_silu"""
    if not FUSED_CONV[0] or getattr(Q, "_q38_conv", False):
        return
    from ttf.q38 import moe_gg
    ref = Q.causal_conv1d_fn

    def f(x, weight, bias=None, activation=None, **kw):
        y = moe_gg.causal_conv1d_silu(x, weight, bias, activation, **kw)
        return ref(x, weight, bias, activation=activation, **kw) if y is None else y
    Q.causal_conv1d_fn = f
    Q._q38_conv = True


def _swiglu(gu, I):
    return F.silu(gu[:, :I]) * gu[:, I:]


# ---- PLE (n-gram embedding), after the reference's hashing (prime head vocabularies, splitmix64 multipliers)
_M64 = (1 << 64) - 1


def _splitmix64(v):
    v = (v + 0x9E3779B97F4A7C15) & _M64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & _M64
    return (v ^ (v >> 31)) & _M64


def _multipliers(vocab, n, ple_index, seed):
    half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
    base = seed + 10007 * ple_index
    return torch.tensor([2 * (_splitmix64((base + 0x9E3779B97F4A7C15 * (i + 1)) & _M64) % half) + 1 for i in range(n)],
                        dtype=torch.long, device="cpu")


def _is_prime(v):
    if v < 2 or v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NGramEmbedding(nn.Module):
    """ids [B, T] -> [B, T, ple_embed_dim]: (ngram_size - 1) x heads_per_ngram hashed heads of dim embed / heads. The
    table (51B entries at the release size) is NOT a parameter: `table` is set from outside (a CPU / mmapped tensor,
    FP8 with `scale`, or bf16); the gather runs where it lives and the rows move to the ids' device."""

    def __init__(self, cfg, ple_index):
        super().__init__()
        self.n, self.hpn = cfg.ngram_size, cfg.heads_per_ngram
        self.heads = (self.n - 1) * self.hpn
        self.dim = cfg.ple_embed_dim // self.heads
        self.eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
        sizes, offs, tot = [], [], 0
        for h in range(self.heads):        # the reference: head g gets the (g + 1)-th prime after base - 1, g global
            sizes.append(_nth_prime_after(cfg.ngram_vocab_size_base - 1, ple_index * self.heads + h + 1))
            offs.append(tot); tot += sizes[-1]
        self.total = tot
        self.rows = math.ceil(tot / cfg.make_ngram_vocab_size_divisible_by) * cfg.make_ngram_vocab_size_divisible_by
        self.register_buffer("mult", _multipliers(cfg.vocab_size, self.n, ple_index, cfg.seed), persistent=False)
        self.register_buffer("sizes", torch.tensor(sizes, dtype=torch.long, device="cpu"), persistent=False)
        self.register_buffer("offs", torch.tensor(offs, dtype=torch.long, device="cpu"), persistent=False)
        self.table, self.scale = None, None

    def _shift(self, ids, s):
        """ids shifted right by s inside each EOS-delimited segment, EOS where the n-gram reaches past its start"""
        if s == 0:
            return ids
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        eos_pos = torch.where(ids == self.eos, pos, -1)
        prev = torch.cat([eos_pos.new_full((B, 1), -1), torch.cummax(eos_pos, 1).values[:, :-1]], 1)
        in_seg = pos[None] - (prev + 1)
        src = pos - s
        sh = ids.gather(1, src.clamp_min(0)[None].expand(B, -1))
        return torch.where((in_seg >= s) & (src[None] >= 0), sh, ids.new_full((), self.eos))

    def indices(self, ids):
        """[B, T, heads] row indices into the table (no cached context: a window starts after an EOS)"""
        ids = ids.long()
        hist = torch.cat([ids.new_full((ids.shape[0], self.n - 1), self.eos), ids], 1)
        sh = [self._shift(hist, s) for s in range(self.n)]
        out = []
        for g in range(2, self.n + 1):
            a, b = (g - 2) * self.hpn, (g - 1) * self.hpn
            mix = sh[0] * self.mult[0]
            for p in range(1, g):
                mix = torch.bitwise_xor(mix, sh[p] * self.mult[p])
            out.append(torch.remainder(mix.unsqueeze(-1), self.sizes[a:b].view(1, 1, -1)) + self.offs[a:b].view(1, 1, -1))
        return torch.cat(out, -1)[:, self.n - 1:]

    def forward(self, ids):
        idx = self.indices(ids)
        flat = idx.reshape(-1)
        if torch.is_tensor(self.table):
            rows = self.table[flat.to(self.table.device)]
        else:                                                   # q38_io.MmapTable (host, on demand)
            rows = self.table[flat]
        rows = rows.to(ids.device).float()
        if self.scale is not None:
            s = self.scale.to(ids.device).float()
            if s.numel() == 1:
                rows = rows * s.reshape(())
            elif s.dim() >= 1 and s.shape[0] >= self.total and s.numel() == s.shape[0]:
                rows = rows * s[flat][:, None]                   # one scale per table row
            else:
                rows = rows * s.reshape(-1)                      # one per column
        return rows.view(*idx.shape[:2], -1)


class PLE(nn.Module):
    def __init__(self, cfg, ple_index):
        super().__init__()
        self.d, self.hc = cfg.hidden_size, cfg.hc_count
        D, E = self.d * self.hc, cfg.ple_embed_dim
        self.ple_embedding = NGramEmbedding(cfg, ple_index)
        self.key_proj = nn.Linear(E, D, bias=False)
        self.value_proj = nn.Linear(E, self.d, bias=False)
        self.norm_key = GroupRMSNorm(D, self.d, cfg.rms_norm_eps)
        self.norm_query = GroupRMSNorm(D, self.d, cfg.rms_norm_eps)
        self.norm_conv = GroupRMSNorm(D, self.d, cfg.rms_norm_eps)
        self.pad = (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size
        self.conv1d = nn.Conv1d(D, D, kernel_size=cfg.ple_conv_kernel_size, groups=D, dilation=cfg.ngram_size, bias=False)

    def forward(self, h, ids):
        e = self.ple_embedding(ids).to(h.dtype)
        k = self.norm_key(self.key_proj(e)).unflatten(-1, (self.hc, self.d))
        v = self.value_proj(e)
        q = self.norm_query(h).unflatten(-1, (self.hc, self.d))
        g = (k * q).sum(-1, keepdim=True) / math.sqrt(self.d)
        g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
        gv = (torch.sigmoid(g) * v.unsqueeze(-2)).flatten(-2)
        c = F.pad(self.norm_conv(gv).transpose(1, 2), (self.pad, 0))
        return gv + F.silu(self.conv1d(c)).transpose(1, 2)


class DenseMat(nn.Module):
    """E float matrices [E, N, K] in the routed layout (the reference and the conversion source); forward(x sorted by
    expert [T, K], offsets [E + 1]) -> [T, N], as TernaryExperts (moe_ttf.py), which replaces it for training"""

    def __init__(self, E, N, K, dtype=torch.bfloat16):
        super().__init__()
        self.E, self.N, self.K = E, N, K
        self.weight = nn.Parameter(torch.empty(E, N, K, dtype=dtype), requires_grad=False)
        self.bias = None             # [E, N] (the conversion's fake-quantised layers)

    def forward(self, x, offsets):
        out = x.new_empty(x.shape[0], self.N)
        o = offsets.tolist()
        for e in range(self.E):
            if o[e + 1] > o[e]:
                out[o[e]:o[e + 1]] = F.linear(x[o[e]:o[e + 1]], self.weight[e].to(x.dtype),
                                              None if self.bias is None else self.bias[e].to(x.dtype))
        return out


class _Rows(torch.autograd.Function):
    """x[idx] for a permutation idx of x's rows, backward g[inv] (inv the inverse permutation): a gather both ways, no
    atomics (x[idx]'s backward is an index_put with accumulate)"""
    @staticmethod
    def forward(ctx, x, idx, inv):
        ctx.save_for_backward(idx, inv)
        return x[idx]

    @staticmethod
    def backward(ctx, g):
        idx, inv = ctx.saved_tensors
        return _Rows.apply(g, inv, idx), None, None


class _Slots(torch.autograd.Function):
    """x [T, d] -> its routed slots in expert order, x[order // k] [T k, d] (one gather; x expanded k times and then
    permuted is the same rows), backward g[inv] summed over each token's k slots (as the expand's backward)"""
    @staticmethod
    def forward(ctx, x, order, inv, k):
        ctx.save_for_backward(inv)
        ctx.k = k
        return x[order // k]

    @staticmethod
    def backward(ctx, g):
        inv, = ctx.saved_tensors
        return g[inv].view(-1, ctx.k, g.shape[-1]).sum(1), None, None, None


FUSED_COMBINE = [True]   # CUDA bf16: the routed slots' weighting, gather and k-sum as moe_gg.Combine


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.E, self.k, self.I = cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size
        self.gate = nn.Linear(cfg.hidden_size, self.E, bias=False)          # the router (float, frozen)
        self.gu = DenseMat(self.E, 2 * self.I, cfg.hidden_size)            # gate rows first, then up
        self.dn = DenseMat(self.E, cfg.hidden_size, self.I)
        self.shared_expert = Q.Qwen3_5MoeMLP(cfg, intermediate_size=cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)
        self.rec = None          # a list: each call's expert offsets appended (the trainer's host counts for the update)
        self.capture = None      # callables (x_sorted, offsets, ...) seeing the routed inputs of gu / dn (conversion)
        self.capture_dn = None

    def route(self, x):
        """x [T, d] -> top-k (weights [T, k] renormalised softmax, experts [T, k])"""
        p = torch.softmax(F.linear(x, self.gate.weight).float(), -1)
        w, e = torch.topk(p, self.k, -1)
        return (w / w.sum(-1, keepdim=True)).to(x.dtype), e

    def forward(self, h):
        B, T, d = h.shape
        x = h.reshape(-1, d)
        w, e = self.route(x)
        flat = e.reshape(-1)
        order = torch.argsort(flat, stable=True)                           # routed slots grouped by expert
        inv = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=x.device))
        offsets = torch.zeros(self.E + 1, dtype=torch.long, device=x.device)
        offsets[1:] = torch.cumsum(torch.zeros(self.E, dtype=torch.long, device=x.device).scatter_add_(
            0, flat, torch.ones_like(flat)), 0)                                 # (bincount syncs on CUDA)
        # the routed slots in expert order: a permutation of the slots (each token k times), so gathers both ways and
        # the k-sum in fp32 (deterministic; an index_add_ in bf16 is atomic and order-dependent)
        if self.rec is not None:
            self.rec.append(offsets)
        xs = _Slots.apply(x, order, inv, self.k)
        if self.capture is not None:
            self.capture(xs, offsets, order, w)
        gu = self.gu(xs, offsets)
        a = _swiglu(gu, self.I)
        if self.capture is not None:
            self.capture_dn(a, offsets)
        ys = self.dn(a, offsets)
        if FUSED_COMBINE[0] and ys.is_cuda and ys.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            from ttf.q38.moe_gg import Combine       # (one kernel each way, no [T k, d] intermediates)
            y = Combine.apply(ys, w, order, inv, self.k).to(x.dtype)
        else:
            ys = ys * w.reshape(-1)[order, None]
            y = _Rows.apply(ys, inv, order).float().view(-1, self.k, d).sum(1).to(x.dtype)   # (gather in bf16: exact)
        y = y + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return y.view(B, T, d)


class Layer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.kind = cfg.layer_types[i]
        if self.kind == "linear_attention":
            self.linear_attn = Q.Qwen3_5MoeGatedDeltaNet(cfg, i)
            self.linear_attn.norm.activation = cfg.output_gate_type or cfg.hidden_act
        else:
            self.self_attn = Q.Qwen3_5MoeAttention(cfg, i)
            ix = nn.Module()                                                # the QSA indexer: kept, unused for T <= budget
            ix.index_qk_proj = nn.Linear(cfg.hidden_size, (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim,
                                         bias=False)
            ix.q_layernorm = GroupRMSNorm(cfg.indexer_head_dim, None, cfg.rms_norm_eps)
            ix.k_layernorm = GroupRMSNorm(cfg.indexer_head_dim, None, cfg.rms_norm_eps)
            self.self_attn.indexer = ix
            self.budget = cfg.indexer_budget
        self.mlp = MoE(cfg)
        pl = cfg.ple_layer_ids or []
        self.ple = PLE(cfg, pl.index(i + 1)) if (i + 1) in pl else None
        self.attn_hyper_connection = GatedResidual(cfg)
        self.mlp_hyper_connection = GatedResidual(cfg)

    def mixer_in(self, h, pe, ids):
        """h after the PLE and the attention / DeltaNet sub-block (the MoE sub-block's input stream)"""
        if self.ple is not None:
            h = h + self.ple(h, ids)
        x, w = self.attn_hyper_connection(h)
        if self.kind == "linear_attention":
            y = self.linear_attn(x)
        else:
            assert x.shape[1] <= self.budget, "QSA indexer not implemented for T > indexer_budget"
            y, _ = self.self_attn(x, pe, None)
        return GatedResidual.inject(h, y, w)

    def forward(self, h, pe, ids):
        h = self.mixer_in(h, pe, ids)
        x, w = self.mlp_hyper_connection(h)
        return GatedResidual.inject(h, self.mlp(x), w)


class Q38(nn.Module):
    def __init__(self, cfg, layers=None):
        super().__init__()
        self.cfg = cfg
        n = cfg.num_hidden_layers if layers is None else layers
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Layer(cfg, i) for i in range(n)])
        self.hyper_connection_mixer = GatedResidual(cfg, combine=False)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = Q.Qwen3_5MoeTextRotaryEmbedding(cfg)

    def embed(self, ids):
        return self.embed_tokens(ids).repeat(1, 1, self.cfg.hc_count)

    def rope(self, T, dev, dtype=torch.bfloat16):
        pos = torch.arange(T, device=dev)[None]
        return self.rotary(torch.empty(1, device=dev, dtype=dtype), pos)

    def head_in(self, h):
        return self.hyper_connection_mixer(h)

    def forward(self, ids):
        h, pe = self.embed(ids), self.rope(ids.shape[1], ids.device, self.embed_tokens.weight.dtype)
        for L in self.layers:
            h = L(h, pe, ids)
        return self.lm_head(self.head_in(h))

    def param_groups(self, i):
        """(trainable float params, frozen float params) of layer i besides the ternary weights (P's trainer)"""
        frozen_keys = ("mlp.gate.", "shared_expert_gate", "hyper_connection", "ple.", "in_proj_a", "in_proj_b",
                       "conv1d", "A_log", "dt_bias", "indexer")
        tr, fr = [], []
        for n, p in self.layers[i].named_parameters():
            if n.startswith("mlp.gu.") or n.startswith("mlp.dn."):
                continue
            (fr if any(k in n for k in frozen_keys) else tr).append((n, p))
        return tr, fr


# ---- checkpoint loading (the FP8 release: experts FP8 e4m3 with 128 x 128 block scales, the rest bf16)
def safetensor_index(path):
    wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    return {k: os.path.join(path, f) for k, f in wm.items()}


def dequant_fp8(w, s_inv, block=128):
    """W = w x s_inv per (block x block) tile"""
    N, K = w.shape
    s = s_inv.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * s


class Reader:
    """lazy tensor reads from the sharded checkpoint, one open handle per shard"""

    def __init__(self, path, dev="cpu"):
        self.idx, self.dev, self.h = safetensor_index(path), dev, {}

    def get(self, name):
        from safetensors import safe_open
        f = self.idx[name]
        if f not in self.h:
            self.h[f] = safe_open(f, "pt", device=self.dev)
        return self.h[f].get_tensor(name)

    def has(self, name):
        return name in self.idx

    def weight(self, name):
        """a float weight, FP8-dequantised when the checkpoint stores it so"""
        w = self.get(name + ".weight")
        if self.has(name + ".weight_scale_inv"):
            return dequant_fp8(w, self.get(name + ".weight_scale_inv"))
        return w


PFX = "model.language_model."


def set_param(root, name, t):
    """replace parameter `name` of `root` (meta or not) by one on tensor t, keeping requires_grad"""
    mod, _, pn = name.rpartition(".")
    m = root.get_submodule(mod) if mod else root
    old = m._parameters[pn]
    m._parameters[pn] = nn.Parameter(t, requires_grad=old.requires_grad)


@torch.no_grad()
def load_layer(L, i, rd, dev, dtype=torch.bfloat16, experts=True):
    """layer i's float weights from the checkpoint into L (on dev); experts=False leaves the routed experts alone"""
    p = f"{PFX}layers.{i}."
    for n, t in list(L.named_parameters()):
        if n.startswith("mlp.gu.") or n.startswith("mlp.dn.") or n.startswith("ple.ple_embedding"):
            continue
        key = p + n
        v = rd.weight(key[:-len(".weight")]) if n.endswith(".weight") else rd.get(key)
        set_param(L, n, v.to(dev, dtype).view(t.shape))
    if L.ple is not None:                          # check our hashing constants against the checkpoint's buffers
        ng, q = L.ple.ple_embedding, p + "ple.ple_embedding."
        for mine, theirs in ((ng.mult, "layer_multipliers"), (ng.sizes, "ngram_heads_vocab_sizes"),
                             (ng.offs, "ngram_heads_offsets")):
            if rd.has(q + theirs):
                assert torch.equal(mine.cpu(), rd.get(q + theirs).cpu().long()), f"PLE {theirs} differs"
    if experts:
        E, I = L.mlp.E, L.mlp.I
        gu = torch.empty(L.mlp.gu.weight.shape, dtype=dtype, device=dev)
        dn = torch.empty(L.mlp.dn.weight.shape, dtype=dtype, device=dev)
        for e in range(E):
            q = f"{p}mlp.experts.{e}."
            gu[e, :I] = rd.weight(q + "gate_proj").to(dev, dtype)
            gu[e, I:] = rd.weight(q + "up_proj").to(dev, dtype)
            dn[e] = rd.weight(q + "down_proj").to(dev, dtype)
        set_param(L, "mlp.gu.weight", gu); set_param(L, "mlp.dn.weight", dn)


def ngram_table(rd, i, parts):
    """the PLE table of layer i: the shard rows concatenated (CPU, as stored: FP8 + a scale), and the scale"""
    p = f"{PFX}layers.{i}.ple.ple_embedding.ngram_embedding."
    T = torch.cat([rd.get(f"{p}shard_{j}.weight") for j in range(parts)], 0)
    s = rd.get(p + "weight_scale") if rd.has(p + "weight_scale") else None
    return T, s


@torch.no_grad()
def load_globals(m, rd, dev, dtype=torch.bfloat16):
    set_param(m, "embed_tokens.weight", rd.get(PFX + "embed_tokens.weight").to(dev, dtype))
    set_param(m, "lm_head.weight", rd.get("lm_head.weight").to(dev, dtype))
    for n, t in list(m.hyper_connection_mixer.named_parameters()):
        set_param(m.hyper_connection_mixer, n, rd.get(PFX + "hyper_connection_mixer." + n).to(dev, dtype))


def build(path_or_cfg, layers=None, device="meta"):
    """the model with its parameters on `device` (meta: empty, filled by load_layer / load_globals); the rotary and
    n-gram buffers are real (CPU; .to(dev) moves them)"""
    cfg = path_or_cfg if isinstance(path_or_cfg, Qwen3_5MoeTextConfig) else text_config(path_or_cfg)
    with torch.device(device):
        m = Q38(cfg, layers)
    m.rotary = Q.Qwen3_5MoeTextRotaryEmbedding(cfg, device="cpu") if device == "meta" else m.rotary
    return m


def ngram_table_memmap(path, i):
    """(table, scale) of layer i's PLE n-gram table for NGramEmbedding: rows read on demand from the checkpoint shards
    (q38_io.MmapTable, page cache), the stored scale (or None)"""
    from ttf.q38 import store as q38_io
    cfg = text_config(path)
    rd = Reader(path)
    k = f"{PFX}layers.{i}.ple.ple_embedding.ngram_embedding.weight_scale"
    return q38_io.MmapTable(path, i, cfg.split_ngram_parts), (rd.get(k).float() if rd.has(k) else None)
