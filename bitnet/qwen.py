"""Qwen3.5 / Qwen3.8 (model_type qwen3_5: Gated DeltaNet + gated full attention) with ternary linears, for --init_hf.

The Hugging Face implementation (transformers >= 5.8, fla kernels for the delta rule) runs the model; every large
linear of the text decoder becomes a KernelTernaryLinear, so TTF's update, the row / column scales and the low-rank
adapter work unchanged. Kept in float: the token embedding and the output head (frozen, bf16: 248k x d), the RMSNorm
gains, the DeltaNet's short convolution, decay and gate parameters (A_log, dt_bias, in_proj_a, in_proj_b: 16-row
projections), the q / k norms. Text only: the vision tower and the multi-token-prediction layer are not built.

The wrapper takes the trainer's interface: forward(idx, targets) -> (None, mean loss), with the head and the
cross-entropy in chunks of loss_chunk tokens (the logits of a 248k vocabulary are never all materialised)."""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# linears kept in float (tiny, or gates whose rounding would be all error)
_KEEP_FLOAT = ("in_proj_a", "in_proj_b")


def is_qwen35(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    return c.get("model_type", "").startswith("qwen3_5") or c.get("text_config", {}).get("model_type", "").startswith("qwen3_5")


def text_config(path):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(path)
    return c.text_config if hasattr(c, "text_config") else c


def model_config(path, max_seq_len=2048):
    """our ModelConfig with the text decoder's sizes (for the trainer's report and bookkeeping only)"""
    from .config import ModelConfig
    t = text_config(path)
    nh = t.num_attention_heads        # Qwen3.8-27B: 24 heads x head_dim 256 != hidden 5120, which ModelConfig rejects;
    if t.hidden_size % nh:            # its heads are unused here (transformers' attention runs), so report the KV heads
        nh = t.num_key_value_heads
    return ModelConfig(vocab_size=t.vocab_size, dim=t.hidden_size, n_layers=t.num_hidden_layers,
                       n_heads=nh, n_kv_heads=t.num_key_value_heads,
                       hidden_dim=t.intermediate_size, max_seq_len=max_seq_len,
                       tie_embeddings=bool(getattr(t, "tie_word_embeddings", False)))


def _text_state(path):
    """the language model's tensors with the 'model.language_model.' / 'model.' prefixes normalised to 'model.'"""
    from .hf_init import load_hf_tensors
    sd = load_hf_tensors(path)
    out = {}
    for k, v in sd.items():
        if k.startswith("model.visual") or k.startswith("mtp") or ".mtp" in k:
            continue
        k2 = k.replace("model.language_model.", "model.")
        out[k2] = v
    return out


class QwenTern(nn.Module):
    loss_chunk = 2048

    def __init__(self, hf_model, grad_checkpoint=True):
        super().__init__()
        self.hf = hf_model
        self.layers = hf_model.model.layers
        self.grad_checkpoint = grad_checkpoint
        self.ckpt_skip = 0
        self.kd_logp = None
        self.kd_w = 1.0
        self.kd_topk = None      # --kd_cache: (ids [T, k] long, log-probs [T, k]) of the teacher's top k per token

    def _head_w(self):
        lm = self.hf.lm_head
        return lm.weight

    def forward(self, idx, targets=None):
        m = self.hf.model
        ckpt = self.grad_checkpoint and self.training
        if ckpt and not m.gradient_checkpointing:
            self.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif not ckpt and m.gradient_checkpointing:
            self.hf.gradient_checkpointing_disable()
        h = m(input_ids=idx, use_cache=False).last_hidden_state
        w = self._head_w()
        if targets is None:
            return F.linear(h, w.to(h.dtype)), None
        hf_, tf = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        n = self.loss_chunk
        total = hf_.new_zeros((), dtype=torch.float32)
        kd = self.kd_logp
        for i in range(0, hf_.shape[0], n):
            hc, tc = hf_[i:i + n], tf[i:i + n]
            if self.kd_topk is not None:
                f_ = _kd_topk_chunk
                args_ = (hc, w.to(hc.dtype), tc, self.kd_topk[0][i:i + n], self.kd_topk[1][i:i + n], self.kd_w)
            elif kd is not None:
                from .model import BitTransformer
                f_ = BitTransformer._loss_chunk_kd
                args_ = (hc, w.to(hc.dtype), tc, kd[i:i + n], self.kd_w)
            else:
                f_ = _ce_chunk
                args_ = (hc, w.to(hc.dtype), tc)
            total = total + (checkpoint(f_, *args_, use_reentrant=False) if ckpt else f_(*args_))
        return None, total / (tf != -1).sum().clamp_min(1)

    def float_tail_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def _ce_chunk(h, w, tgt):
    return F.cross_entropy(F.linear(h, w).float(), tgt, ignore_index=-1, reduction="sum")


def _kd_topk_chunk(h, w, tgt, ids, tlp, a):
    """cross-entropy to the teacher's top-k distribution (renormalised over its k tokens; the gradient of the KL to it),
    mixed with next-token cross-entropy: (1 - a) CE + a KD, summed over tokens"""
    z = F.linear(h, w).float()
    lse = torch.logsumexp(z, -1, keepdim=True)        # no full log-softmax: the k gathered logits minus the logsumexp
    kd = -(torch.softmax(tlp.float(), -1) * (z.gather(1, ids) - lse)).sum()
    if a >= 1.0:
        return kd
    ok = tgt != -1
    ce = -(z.gather(1, tgt.clamp_min(0)[:, None]) - lse)[:, 0][ok].sum()
    return (1 - a) * ce + a * kd


def _swap_linears(hf_model, make_linear):
    """replace the decoder's nn.Linear modules (except _KEEP_FLOAT) by make_linear(in, out); returns {name: module}"""
    swapped = {}
    for name, mod in list(hf_model.model.layers.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname not in _KEEP_FLOAT:
                assert child.bias is None, f"{name}.{cname} has a bias"
                new = make_linear(child.in_features, child.out_features)
                setattr(mod, cname, new)
                swapped[f"model.layers.{name}.{cname}" if name else f"model.layers.{cname}"] = new
    return swapped


def build_qwen(path, mode, make_linear=None, grad_checkpoint=True, device="cuda", dtype=torch.bfloat16):
    """the text decoder on `device`: mode 'fp32' (all linears float, the reference) or 'kernel' (make_linear for the large
    linears); float parts in `dtype`. Weights are NOT loaded here (load_qwen)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    t = text_config(path)
    t._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = Qwen3_5ForCausalLM(t)
    shapes = {}
    if mode == "kernel":    # the large linears are dropped before memory is allocated (27B: 50 GB of bf16 never made)
        for name, mod in list(hf.model.layers.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and cname not in _KEEP_FLOAT:
                    shapes[(name, cname)] = (child.in_features, child.out_features)
                    setattr(mod, cname, nn.Identity())
    hf = hf.to_empty(device=device)
    for p in hf.parameters():
        p.data = p.data.to(dtype)
    # the ternary layers are built after to_empty (which would leave their own buffers uninitialised), in fp32
    swapped = {}
    for (name, cname), (i, o) in shapes.items():
        mod = hf.model.layers.get_submodule(name) if name else hf.model.layers
        new = make_linear(i, o).to(device)
        setattr(mod, cname, new)
        swapped[f"model.layers.{name}.{cname}" if name else f"model.layers.{cname}"] = new
    # the rotary module is rebuilt: to_empty leaves its inv_freq buffer uninitialised (non-persistent, not in checkpoints)
    hf.model.rotary_emb = type(hf.model.rotary_emb)(t).to(device)
    m = QwenTern(hf, grad_checkpoint=grad_checkpoint)
    return m, swapped


@torch.no_grad()
def load_qwen(m, path, swapped, method="als", rc=True, hess=None, log=print):
    """copy the float tensors; convert each swapped linear's weight to trits + row / column scales (hf_init.ternarize /
    ternarize_gptq); embedding and head are frozen"""
    from .hf_init import ternarize, ternarize_gptq, _rel_err, _out_err
    from .kernel import pack_rows, trit_beta
    sd = _text_state(path)
    hf = m.hf
    own = dict(hf.named_parameters())
    sw = {k + ".weight" for k in swapped}
    missing = []
    for k, p in own.items():
        if k in sd:
            p.copy_(sd[k].to(p.dtype))
        elif k == "lm_head.weight" and "model.embed_tokens.weight" in sd:
            p.copy_(sd["model.embed_tokens.weight"].to(p.dtype))
        elif not any(k.startswith(s.rsplit(".weight", 1)[0]) for s in sw):
            missing.append(k)
    if missing:
        log(f"load_qwen: {len(missing)} float tensors not in the checkpoint (kept at init): {missing[:6]}")
    hf.model.embed_tokens.weight.requires_grad_(False); hf.lm_head.weight.requires_grad_(False)
    errs, oerrs = [], []
    for name, lin in swapped.items():
        W = sd[name + ".weight"].to(lin.wpacked.device).float()
        T, a, c = ternarize(W, "als" if method == "gptq" else method)
        if hess is not None:
            Hl = hess[name].to(W.device)
            e0 = _out_err(W, T, a, c, Hl)
            if method == "gptq":
                T, a, c = ternarize_gptq(W, Hl, a, c)
            oerrs.append((e0, _out_err(W, T, a, c, Hl))); del Hl
        errs.append(_rel_err(W, T, a, c))
        lin.wpacked.copy_(pack_rows(T))
        beta = float(trit_beta(lin.wpacked, lin.K)) if lin.use_beta else 1.0
        if rc and lin.row_scale is not None:
            lin.row_scale.copy_(a / beta); lin.col_scale.copy_(c)
        else:
            raise SystemExit("Qwen conversion needs --rc_scale")
        del W
    if errs:
        log(f"--init_hf {path}: {method}, {len(errs)} ternary linears, relative weight error mean "
            f"{sum(errs) / len(errs):.4f} (max {max(errs):.4f})")
    if oerrs:
        e0, e1 = zip(*oerrs)
        log(f"  output-space error mean {sum(e1) / len(e1):.4f} (ALS rounding {sum(e0) / len(e0):.4f})")


@torch.no_grad()
def calib_hessians_qwen(path, data_path, n_seq=64, seq=2048, dev="cuda", dtype_u32=True, log=print):
    """input second moments X^T X / tokens of every large linear in the full-precision model ({linear name: H})"""
    import numpy as np
    m, _ = build_qwen(path, "fp32", grad_checkpoint=False, device=dev)
    sd = _text_state(path)
    for k, p in m.hf.named_parameters():
        src = sd.get(k, sd.get("model.embed_tokens.weight") if k == "lm_head.weight" else None)
        if src is not None:
            p.data.copy_(src.to(p.dtype))
    del sd
    m.eval()
    H, cnt, hooks = {}, {}, []
    for name, mod in m.hf.model.layers.named_modules():
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] not in _KEEP_FLOAT:
            key = "model.layers." + name
            H[key] = torch.zeros(mod.in_features, mod.in_features, device=dev); cnt[key] = 0

            def hook(mod_, inp, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[key].addmm_(x.T, x); cnt[key] += x.shape[0]
            hooks.append(mod.register_forward_pre_hook(hook))
    data = np.memmap(data_path, dtype=np.uint32 if dtype_u32 else np.uint16, mode="r")
    rng = np.random.default_rng(4321)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for s in rng.integers(0, len(data) - seq - 1, n_seq):
            x = torch.tensor(data[s:s + seq].astype(np.int64), device=dev)[None]
            m(x, x)
    for h in hooks:
        h.remove()
    out = {k: (v / max(cnt[k], 1)).cpu() for k, v in H.items()}
    del m, H; torch.cuda.empty_cache()
    log(f"calibration: input second moments of {len(out)} linears from {n_seq} x {seq} tokens")
    return out


class _StopFwd(Exception):
    pass


@torch.no_grad()
def layer_inputs(m, windows, mb, dev):
    """the first decoder layer's inputs for the windows (micro-batches of mb) and the keyword arguments transformers
    passes to EACH layer (rotary embeddings; the full-attention and the recurrent masks differ by layer type), taken
    from one full forward of the first micro-batch; the other micro-batches stop at layer 0"""
    import numpy as np
    layers = m.hf.model.layers
    KW, H, hooks = [None] * len(layers), [], []

    def mk(i):
        def pre(mod, args, kwargs):
            if KW[i] is None:
                KW[i] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            if i == 0:
                H.append((args[0] if args else kwargs["hidden_states"]).detach())
                if all(k is not None for k in KW):
                    raise _StopFwd
        return pre
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_pre_hook(mk(i), with_kwargs=True))
    for b in range(0, len(windows), mb):
        x = torch.tensor(np.stack([w.astype(np.int64) for w in windows[b:b + mb]]), device=dev)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                m.hf.model(input_ids=x, use_cache=False)
        except _StopFwd:
            pass
    for h in hooks:
        h.remove()
    return H, KW


def _safetensor_index(path):
    """{tensor name (normalised to 'model.'): file} for lazy per-layer loading"""
    import glob
    idx = os.path.join(path, "model.safetensors.index.json")
    files = {}
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        for k, f in wm.items():
            files[k.replace("model.language_model.", "model.")] = (k, os.path.join(path, f))
    else:
        from safetensors import safe_open
        for f in glob.glob(os.path.join(path, "*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    files[k.replace("model.language_model.", "model.")] = (k, f)
    return files


def _get(files, name, dev):
    from safetensors import safe_open
    k, f = files[name]
    with safe_open(f, "pt", device="cpu") as h:
        return h.get_tensor(k).to(dev)


@torch.no_grad()
def convert_qwen_streaming(m, path, swapped, data_path, method="gptq", n_seq=64, seq=2048, mb=4, dev="cuda",
                           dtype_u32=True, log=print):
    """sequential, layer-by-layer conversion for models that do not fit as a whole in full precision (27B): the float
    tensors are loaded first; then for each decoder layer in turn its full-precision linears are put in place (read
    lazily from the safetensors), the calibration activations (inputs as produced by the already converted layers below)
    are run through it to collect each linear's input second moment, the linears are rounded (GPTQ or ALS) into the
    ternary layers, and the activations go on through the converted layer. Peak: the model's ternary form + one layer
    in full precision + the calibration activations (n_seq x seq x d bf16)."""
    import numpy as np
    from .hf_init import ternarize, ternarize_gptq, _rel_err, _out_err
    from .kernel import pack_rows, trit_beta
    files = _safetensor_index(path)
    hf = m.hf
    sw = {k + ".weight" for k in swapped}
    for k, p in hf.named_parameters():                  # the float parts
        src = k if k in files else ("model.embed_tokens.weight" if k == "lm_head.weight" else None)
        if src in files and k not in sw:
            p.copy_(_get(files, src, dev).to(p.dtype))
    hf.model.embed_tokens.weight.requires_grad_(False); hf.lm_head.weight.requires_grad_(False)
    data = np.memmap(data_path, dtype=np.uint32 if dtype_u32 else np.uint16, mode="r")
    rng = np.random.default_rng(4321)
    starts = rng.integers(0, len(data) - seq - 1, n_seq)
    layers = hf.model.layers
    H_in, KW = layer_inputs(m, [data[s:s + seq] for s in starts], mb, dev)
    log(f"streaming conversion: {len(layers)} layers, {n_seq} x {seq} calibration tokens in {len(H_in)} micro-batches")
    errs, oerrs = [], []
    for li, layer in enumerate(layers):
        pre_ = f"model.layers.{li}."
        mine = {n: l for n, l in swapped.items() if n.startswith(pre_)}
        fp = {}                                         # this layer's full-precision linears, in place of the ternary ones
        for n, lin in mine.items():
            W = _get(files, n + ".weight", dev)
            f = nn.Linear(W.shape[1], W.shape[0], bias=False, device=dev, dtype=torch.bfloat16); f.weight.copy_(W)
            parent = layer.get_submodule(n[len(pre_):].rsplit(".", 1)[0]); setattr(parent, n.rsplit(".", 1)[1], f)
            fp[n] = (f, parent)
        Hs = {n: torch.zeros(f.in_features, f.in_features, device=dev) for n, (f, _) in fp.items()}
        cnt = {n: 0 for n in fp}
        hooks = []
        for n, (f, _) in fp.items():
            def hook(mod_, inp, n=n):
                xx = inp[0].reshape(-1, inp[0].shape[-1]).float(); Hs[n].addmm_(xx.T, xx); cnt[n] += xx.shape[0]
            hooks.append(f.register_forward_pre_hook(hook))
        kw = KW[li]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for h in H_in:
                layer(h, **kw)
        for hh in hooks:
            hh.remove()
        for n, (f, parent) in fp.items():               # round into the ternary layer, put it back
            lin = mine[n]
            W = f.weight.float(); Hl = Hs.pop(n) / max(cnt[n], 1)
            T, a, c = ternarize(W, "als")
            e0 = _out_err(W, T, a, c, Hl)
            if method == "gptq":
                T, a, c = ternarize_gptq(W, Hl, a, c)
            oerrs.append((e0, _out_err(W, T, a, c, Hl))); errs.append(_rel_err(W, T, a, c))
            lin.wpacked.copy_(pack_rows(T))
            beta = float(trit_beta(lin.wpacked, lin.K)) if lin.use_beta else 1.0
            lin.row_scale.copy_(a / beta); lin.col_scale.copy_(c)
            setattr(parent, n.rsplit(".", 1)[1], lin)
            del W, Hl, T
        del fp, Hs
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # on through the converted layer
            H_in = [(lambda o: o[0] if isinstance(o, tuple) else o)(layer(h, **kw)) for h in H_in]
        torch.cuda.empty_cache()
        if li % 8 == 0 or li == len(layers) - 1:
            e0, e1 = zip(*oerrs[-len(mine):])
            log(f"  layer {li}: {len(mine)} linears, output-space error {sum(e1) / len(e1):.4f} "
                f"(ALS {sum(e0) / len(e0):.4f}), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    e0, e1 = zip(*oerrs)
    log(f"--init_hf {path}: {method} (sequential, streaming), {len(errs)} ternary linears, weight error "
        f"{sum(errs) / len(errs):.4f}, output-space error {sum(e1) / len(e1):.4f} (ALS {sum(e0) / len(e0):.4f})")
