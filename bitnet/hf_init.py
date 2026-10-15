"""Start a run from a pretrained full-precision Llama checkpoint (Hugging Face format): --init_hf DIR.

The float parts (token embedding, output head, RMSNorm gains) are copied. Each linear is converted for the run's mode:
  fp32    the weight as is (sanity check: the loss should match the original model's)
  master  the latent is the weight (BitNet's absmean ternary in the forward, as published conversions do)
  kernel  the weight is ternarized (--init_hf_method):
            gptqc    GPTQ on the input covariance instead of the second moment, the layer computing W_hat (x - mu) + W mu:
                     the input mean mu (fixed) is taken out before the activation quantisation and its product with
                     the fp weight is a float bias (trained); for fp models whose residual stream carries a large
                     constant offset (massive activations)
            absmean  T = round(clip(W / mean|W|)) per tensor, the output scale mean|W| (BitNet b1.58)
            row      a per-row scale a_i chosen from a grid to minimise |W_i - a_i T_i| (least-squares magnitude)
            als      W ~ a_i c_j T_ij, row and column scales by alternating least squares (needs --rc_scale)
            scratch  nothing loaded, only the shape (the from-scratch baseline for the same model)
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
        try:
            from safetensors.torch import load_file
        except ImportError:
            load_file = _read_safetensors
        for f in files:
            sd.update(load_file(os.path.join(path, f)))
    else:
        for f in sorted(f for f in os.listdir(path) if f.startswith("pytorch_model") and f.endswith(".bin")):
            sd.update(torch.load(os.path.join(path, f), map_location="cpu", weights_only=True))
    return {k: v.float() for k, v in sd.items()}


_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
              "I32": torch.int32, "U8": torch.uint8, "I8": torch.int8, "BOOL": torch.bool}


def _read_safetensors(fn):
    """safetensors without the package (Myriad's image lacks it): 8-byte little-endian header size, a JSON header
    {name: {dtype, shape, data_offsets}}, then the raw little-endian data"""
    with open(fn, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        buf = bytearray(f.read())
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        dt = _ST_DTYPES[v["dtype"]]; a, b = v["data_offsets"]
        out[k] = torch.frombuffer(buf, dtype=dt, count=(b - a) // torch.empty((), dtype=dt).element_size(),
                                  offset=a).reshape(v["shape"]).clone()
    return out


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


def _out_err(W, T, a, c, H):
    """output-space relative error tr(E H E^T) / tr(W H W^T), E = W - a c T, H = X^T X of the layer's inputs"""
    E = W - a[:, None] * c[None, :] * T.float()
    return float(((E @ H) * E).sum() / ((W @ H) * W).sum())


@torch.no_grad()
def calib_hessians(path, data_path, n_seq=64, seq=2048, dev="cuda", log=print):
    """H = X^T X / tokens of every linear's input in the full-precision model on n_seq random windows of the run's
    data (q / k / v share one, gate / up another): {(block, name): H [K, K] fp32 on the CPU}"""
    import numpy as np
    import torch.nn as nn
    from .model import BitTransformer
    mc = model_config_from_hf(path, max_seq_len=seq)
    m = BitTransformer(mc, grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev)
    init_from_hf(m, path, "fp32", log=lambda *_: None)
    m.eval()
    H, hooks, cnt, MU = {}, [], {}, {}
    own = {"wq": "wq", "wk": "wq", "wv": "wq", "wo": "wo", "w_gate": "w_gate", "w_up": "w_gate", "w_down": "w_down"}
    for i, blk in enumerate(m.layers):
        for sub, name in (("attn", "wq"), ("attn", "wo"), ("ffn", "w_gate"), ("ffn", "w_down")):
            lin = getattr(getattr(blk, sub), name)
            key = (i, name)
            H[key] = torch.zeros(lin.in_features, lin.in_features, device=dev)
            MU[key] = torch.zeros(lin.in_features, device=dev)
            cnt[key] = 0

            def hook(mod, inp, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[key].addmm_(x.T, x); MU[key] += x.sum(0); cnt[key] += x.shape[0]
            hooks.append(lin.register_forward_pre_hook(hook))
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(4321)                   # not the training stream's windows
    for s in rng.integers(0, len(data) - seq - 1, n_seq):
        x = torch.tensor(data[s:s + seq + 1].astype(np.int64), device=dev)[None]
        m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    out = {}
    for (i, name), h in H.items():
        h /= max(cnt[(i, name)], 1)
        mu = MU[(i, name)] / max(cnt[(i, name)], 1)
        for n2, o in own.items():
            if o == name:
                out[(i, n2)] = h.cpu()
                out[("mu", i, n2)] = mu.cpu()       # the input mean (gptqc: centred conversion)
    del m, H
    torch.cuda.empty_cache()
    log(f"calibration: input second moments of {len(out)} linears from {n_seq} x {seq} tokens of {data_path}")
    return out


@torch.no_grad()
def ternarize_gptq(W, H, a, c, block=128, damp=0.01, act_order=True, refits=2):
    """GPTQ (Frantar et al. 2022) onto the ternary grid {-a_i, 0, a_i} c_j: columns are rounded one at a time and each
    one's rounding error is pushed onto the not-yet-rounded columns through the inverse input second moment, so the
    trits minimise the layer's output error |X (W - a c T)^T| instead of |W - a c T|. The column scale c comes from
    ALS (folded into the inputs), the row magnitude a is refitted in output space after each pass."""
    N, K = W.shape
    Wc = W / c[None, :]                                  # fold c into the input: X' = X c, H' = c H c
    Hc = H.to(W.device) * c[:, None] * c[None, :]
    dead = torch.diagonal(Hc) <= 0
    Hc[dead, dead] = 1.0
    Wc[:, dead] = 0.0
    Hc += damp * torch.diagonal(Hc).mean() * torch.eye(K, device=W.device)
    perm = torch.argsort(torch.diagonal(Hc), descending=True) if act_order else torch.arange(K, device=W.device)
    inv = torch.argsort(perm)
    Wp, Hp = Wc[:, perm], Hc[perm][:, perm]
    L = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)          # GPTQ's upper factor of H^-1
    del L, Hinv
    for _ in range(refits):
        Q = Wp.clone()
        T = torch.zeros(N, K, device=W.device)
        ai = a[:, None]
        for i1 in range(0, K, block):
            i2 = min(i1 + block, K)
            Wb = Q[:, i1:i2].clone(); Ub = U[i1:i2, i1:i2]; E = torch.zeros_like(Wb)
            for j in range(i2 - i1):
                w = Wb[:, j]
                t = (w / ai[:, 0]).round().clamp(-1, 1)
                T[:, i1 + j] = t
                e = (w - ai[:, 0] * t) / Ub[j, j]
                Wb[:, j:] -= e[:, None] * Ub[j, j:][None, :]
                E[:, j] = e
            Q[:, i2:] -= E @ U[i1:i2, i2:]
        # output-space least-squares magnitude per row: a_i = <W_i H, T_i> / <T_i H, T_i>
        TH = T @ Hp
        a = ((Wp @ Hp) * T).sum(1) / (TH * T).sum(1).clamp_min(1e-30)
        a = torch.where(a > 0, a, torch.full_like(a, 1e-3))
        del TH, Q
    return T[:, inv].to(torch.int8), a, c


@torch.no_grad()
def init_from_hf(model, path, mode, method="als", rc=False, acc=0.0, theta=16.0, rank=0, clip=0.5, log=print, hess=None):
    """copy / convert the checkpoint into model (built with model_config_from_hf); returns {layer: (U, V)} for
    --init_acc (empty otherwise). method gptq needs hess = calib_hessians(...)"""
    from .flip import KernelTernaryLinear
    from .kernel import pack_rows, trit_beta
    from .master import MasterTernaryLinear
    if method == "scratch":       # the checkpoint's shape only, random init (the baseline for the same model)
        log(f"--init_hf {path}: scratch (shape only, weights not loaded)")
        return {}
    sd = load_hf_tensors(path)
    c_ = model.c
    dev = next(model.parameters()).device
    model.tok_emb.weight.copy_(sd["model.embed_tokens.weight"])
    model.norm.weight.copy_(sd["model.norm.weight"])
    if model.lm_head is not None:
        model.lm_head.weight.copy_(sd.get("lm_head.weight", sd["model.embed_tokens.weight"]))
    acc_state, errs, kept, oerrs = {}, [], [], []
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
            T, a, c = ternarize(W, "als" if method in ("gptq", "gptqc") else method)
            if hess is not None and method != "zero":
                Hl = hess[(i, name)].to(dev)
                e0 = _out_err(W, T, a, c, Hl)
                if method == "gptq":
                    T, a, c = ternarize_gptq(W, Hl, a, c)
                elif method == "gptqc":     # centred: GPTQ on the input covariance; the layer subtracts the input mean
                    mu = hess[("mu", i, name)].to(dev)  # before its activation quantisation and adds W mu back as a bias
                    T, a, c = ternarize_gptq(W, Hl - torch.outer(mu, mu), a, c)
                    lin.enable_centering(mu, W @ mu)
                oerrs.append((e0, _out_err(W, T, a, c, Hl)))
                del Hl
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
    if oerrs:
        e0, e1 = zip(*oerrs)
        log(f"--init_hf {method}: output-space relative error |X (W - a c T)^T|^2 / |X W^T|^2 mean {sum(e1) / len(e1):.4f}"
            f" (als rounding {sum(e0) / len(e0):.4f}) over {len(e1)} layers")
    if kept:
        log(f"--init_acc {acc}: rank {rank} keeps {sum(kept) / len(kept):.3f} of the accumulator's energy on average "
            f"(min {min(kept):.3f})")
    return acc_state
