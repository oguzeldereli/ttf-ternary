"""Start a run from a pretrained full-precision Llama checkpoint (Hugging Face format): --init_hf DIR.

The float parts (token embedding, output head, RMSNorm gains) are copied. Each linear is converted for the run's mode:
  fp32    the weight as is (sanity check: the loss should match the original model's)
  master  the latent is the weight (BitNet's absmean ternary in the forward, as published conversions do)
  kernel  the weight is ternarized (--init_hf_method):
            absmean  T = round(clip(W / mean|W|)) per tensor, the output scale mean|W| (BitNet b1.58)
            row      a per-row scale a_i chosen from a grid to minimise |W_i - a_i T_i| (least-squares magnitude)
            als      W ~ a_i c_j T_ij, row and column scales by alternating least squares (needs --rc_scale)
            zero     every trit 0; the weight goes into the accumulator only (needs --ts and --init_acc), so the first
                     update's firing does the rounding
          the scales go into row_scale / col_scale (with --rc_scale) over the layer's beta = 1 / sqrt(K rho);
          without --rc_scale only the per-tensor part is kept (row and column factors are averaged out).
--init_acc KAPPA (with --ts): the accumulator starts at each weight's position inside its bin, A = KAPPA 2 theta (x - T)
with x = W / (a_i c_j) in trit units (x - T in [-1/2, 1/2]; a weight on a rounding boundary starts at |A| = KAPPA theta),
kept at the run's rank by a truncated SVD. --init_acc_clip C caps |x - T| at C (default 1/2: drops the outward
excess of weights far beyond their +-1 bin; 0 keeps it, so a large weight starts with evidence against being moved in). With --init_hf_method zero, A = KAPPA 2 theta x (all of the weight).

HF Llama stores q / k rows permuted for rotate_half RoPE; this model rotates interleaved pairs (Meta layout), so the
q / k rows are permuted back."""
import json
import os

import torch

_LIN = {"self_attn.q_proj": ("attn", "wq"), "self_attn.k_proj": ("attn", "wk"), "self_attn.v_proj": ("attn", "wv"),
        "self_attn.o_proj": ("attn", "wo"), "mlp.gate_proj": ("ffn", "w_gate"), "mlp.up_proj": ("ffn", "w_up"),
        "mlp.down_proj": ("ffn", "w_down")}


def hf_config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def model_config_from_hf(path, max_seq_len=2048):
    from .config import ModelConfig
    h = hf_config(path)
    return ModelConfig(vocab_size=h["vocab_size"], dim=h["hidden_size"], n_layers=h["num_hidden_layers"],
                       n_heads=h["num_attention_heads"], n_kv_heads=h.get("num_key_value_heads", h["num_attention_heads"]),
                       hidden_dim=h["intermediate_size"], max_seq_len=max_seq_len,
                       rope_theta=float(h.get("rope_theta", 10000.0)), norm_eps=float(h.get("rms_norm_eps", 1e-5)),
                       tie_embeddings=bool(h.get("tie_word_embeddings", False)))


def load_hf_tensors(path):
    """all tensors of the checkpoint (safetensors shards, else pytorch_model*.bin), fp32 on the CPU"""
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    sd = {}
    if files:
        from safetensors.torch import load_file
        for f in files:
            sd.update(load_file(os.path.join(path, f)))
    else:
        for f in sorted(f for f in os.listdir(path) if f.startswith("pytorch_model") and f.endswith(".bin")):
            sd.update(torch.load(os.path.join(path, f), map_location="cpu", weights_only=True))
    return {k: v.float() for k, v in sd.items()}


def _unpermute(w, n_heads):
    """HF rotate_half layout -> interleaved pairs (inverse of transformers' convert_llama_weights_to_hf permute)"""
    out, k = w.shape
    return w.view(n_heads, 2, out // n_heads // 2, k).transpose(1, 2).reshape(out, k)


@torch.no_grad()
def _row_grid(W, grid=torch.linspace(0.3, 2.0, 35)):
    """per-row scale a_i and trits T minimising |W_i - a_i T_i|^2: for each candidate threshold-scale s = g mean|W_i|,
    T = round(clip(W / s)) and the best magnitude a = <W, T> / <T, T>; keep the best per row"""
    m = W.abs().mean(1, keepdim=True).clamp_min(1e-12)
    best_err = torch.full((W.shape[0], 1), float("inf"), device=W.device)
    best_s = m.clone()
    for g in grid.tolist():
        T = (W / (g * m)).round_().clamp_(-1, 1)
        nt = (T * T).sum(1, keepdim=True).clamp_min(1)
        a = (W * T).sum(1, keepdim=True) / nt
        err = (W * W).sum(1, keepdim=True) - a * a * nt          # |W - a T|^2 at the optimal a
        better = err < best_err
        best_err = torch.where(better, err, best_err); best_s = torch.where(better, g * m, best_s)
    T = (W / best_s).round_().clamp_(-1, 1)
    a = ((W * T).sum(1, keepdim=True) / (T * T).sum(1, keepdim=True).clamp_min(1)).clamp_min(1e-12)
    return a.squeeze(1), T


@torch.no_grad()
def ternarize(W, method, iters=4):
    """W [N, K] -> (T int8, row a [N], col c [K]) with W ~ a_i c_j T_ij"""
    N, K = W.shape
    c = torch.ones(K, device=W.device)
    if method == "zero":
        return torch.zeros(N, K, dtype=torch.int8, device=W.device), W.abs().mean(1), c
    if method == "absmean":
        g = W.abs().mean().clamp_min(1e-12)
        T = (W / g).round_().clamp_(-1, 1)
        return T.to(torch.int8), torch.full((N,), float(g), device=W.device), c
    a, T = _row_grid(W)
    if method == "als":
        for _ in range(iters):
            num = (W * T * a[:, None]).sum(0); den = (T * T * (a * a)[:, None]).sum(0)
            c = torch.where(den > 0, num / den.clamp_min(1e-30), c).clamp_min(1e-3)
            c = c / c.mean()
            a, T = _row_grid(W / c[None, :])
    return T.to(torch.int8), a, c


def _rel_err(W, T, a, c):
    return float((W - a[:, None] * c[None, :] * T.float()).norm() / W.norm())


@torch.no_grad()
def init_from_hf(model, path, mode, method="als", rc=False, acc=0.0, theta=16.0, rank=0, clip=0.5, log=print):
    """copy / convert the checkpoint into model (built with model_config_from_hf); returns {layer: (U, V)} for
    --init_acc (empty otherwise)"""
    from .flip import KernelTernaryLinear
    from .kernel import pack_rows, trit_beta
    from .master import MasterTernaryLinear
    sd = load_hf_tensors(path)
    c_ = model.c
    dev = next(model.parameters()).device
    model.tok_emb.weight.copy_(sd["model.embed_tokens.weight"])
    model.norm.weight.copy_(sd["model.norm.weight"])
    if model.lm_head is not None:
        model.lm_head.weight.copy_(sd.get("lm_head.weight", sd["model.embed_tokens.weight"]))
    acc_state, errs, kept = {}, [], []
    for i, blk in enumerate(model.layers):
        p = f"model.layers.{i}."
        blk.attn_norm.weight.copy_(sd[p + "input_layernorm.weight"])
        blk.ffn_norm.weight.copy_(sd[p + "post_attention_layernorm.weight"])
        for hf, (sub, name) in _LIN.items():
            W = sd.pop(p + hf + ".weight").to(dev)
            if name == "wq":
                W = _unpermute(W, c_.n_heads)
            elif name == "wk":
                W = _unpermute(W, c_.n_kv_heads)
            lin = getattr(getattr(blk, sub), name)
            if isinstance(lin, MasterTernaryLinear):
                lin.weight.copy_(W)
                continue
            if not isinstance(lin, KernelTernaryLinear):        # fp32 mode: nn.Linear
                lin.weight.copy_(W)
                continue
            T, a, c = ternarize(W, method)
            if method != "zero":
                errs.append(_rel_err(W, T, a, c))
            lin.wpacked.copy_(pack_rows(T))
            beta = float(trit_beta(lin.wpacked, lin.K)) if lin.use_beta else 1.0
            if method == "zero":     # the trits are all 0, so beta is that of an empty layer; use the density the
                beta = 1.0 / (lin.K * 0.6) ** 0.5   # first firing will give (~0.6 nonzero), refined as trits move
            if rc and lin.row_scale is not None:
                lin.row_scale.copy_(a / beta); lin.col_scale.copy_(c)
            elif lin.row_scale is not None:
                lin.row_scale.fill_(float(a.mean()) / beta); lin.col_scale.fill_(1.0)
            else:
                raise SystemExit("--init_hf on kernel mode needs --rc_scale (the layer has no other scale to hold the "
                                 "weight's magnitude)")
            if acc:
                X = W / (a[:, None] * c[None, :])               # trit units
                D_ = X - T.float()        # only a +-1 trit's outward excess can pass 1/2 (it cannot move further out)
                if clip > 0 and method != "zero":
                    D_.clamp_(-clip, clip)
                A = (acc * 2 * theta) * (D_ if method != "zero" else X); del D_
                r = min(rank or min(lin.N, lin.K), lin.N, lin.K)
                if r < min(lin.N, lin.K):
                    Uf, S, Vh = torch.linalg.svd(A, full_matrices=False)
                    V = Vh[:r].T.contiguous(); U = Uf[:, :r] * S[:r]
                    kept.append(float((S[:r] ** 2).sum() / (S ** 2).sum()))
                    del Uf, S, Vh
                else:
                    Q, _ = torch.linalg.qr(A.T)       # full rank: V orthonormal K x N(<=K), U = A V
                    V = Q[:, :r].contiguous(); U = A @ V; kept.append(1.0)
                acc_state[lin] = (U.contiguous(), V)
                del A, X
            del W
    if errs:
        log(f"--init_hf {path}: {method}, relative error |W - a c T| / |W| mean {sum(errs) / len(errs):.4f} "
            f"(min {min(errs):.4f}, max {max(errs):.4f}) over {len(errs)} layers")
    if kept:
        log(f"--init_acc {acc}: rank {rank} keeps {sum(kept) / len(kept):.3f} of the accumulator's energy on average "
            f"(min {min(kept):.3f})")
    return acc_state
