"""A pretrained full-precision Llama checkpoint (Hugging Face format), --arch-hf DIR: its shape (model_config_from_hf)
and, for --mode fp32 / bitnet and the --teacher, its weights (load_hf_fp). The float parts (token embedding, output
head, RMSNorm gains) and each linear's weight are copied as they are (bitnet: as the master weights, BitNet's absmean
ternary in the forward). Converting the weights to trits and scales for --mode ttf is the ttf-conversion package's
(since 28 Oct 2026; before, ttf/hf_init.py here).

HF Llama stores q / k rows permuted for rotate_half RoPE; this model rotates interleaved pairs (Meta layout), so the
q / k rows are permuted back."""
import json
import os

import torch
import torch.nn as nn

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
    """safetensors without the package (some cluster images lack it): 8-byte little-endian header size, a JSON header
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
def load_hf_fp(model, path, log=print):
    """copy the checkpoint's weights into model (built with model_config_from_hf; its linears nn.Linear or
    MasterTernaryLinear)"""
    from .master import MasterTernaryLinear
    sd = load_hf_tensors(path)
    c_ = model.c
    dev = next(model.parameters()).device
    model.tok_emb.weight.copy_(sd["model.embed_tokens.weight"])
    model.norm.weight.copy_(sd["model.norm.weight"])
    if model.lm_head is not None:
        model.lm_head.weight.copy_(sd.get("lm_head.weight", sd["model.embed_tokens.weight"]))
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
            if not isinstance(lin, (nn.Linear, MasterTernaryLinear)):
                raise SystemExit(f"load_hf_fp: {p}{hf} is a {type(lin).__name__}; ternary layers come from a conversion "
                                 f"(the ttf-conversion package)")
            lin.weight.copy_(W)
            del W
    log(f"--arch-hf {path}: full-precision weights loaded ({len(model.layers)} layers)")
