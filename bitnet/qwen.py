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
    return ModelConfig(vocab_size=t.vocab_size, dim=t.hidden_size, n_layers=t.num_hidden_layers,
                       n_heads=t.num_attention_heads, n_kv_heads=t.num_key_value_heads,
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
            if kd is not None:
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
    hf = hf.to_empty(device=device)
    for p in hf.parameters():
        p.data = p.data.to(dtype)
    # the ternary layers are built after to_empty (which would leave their own buffers uninitialised), in fp32
    swapped = _swap_linears(hf, lambda i, o: make_linear(i, o).to(device)) if mode == "kernel" else {}
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
