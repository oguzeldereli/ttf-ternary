"""Export one of our fp32 checkpoints (mode fp32, e.g. checkpoints/fp32_baseline) as a Hugging Face Llama directory, so
the trainer's --init_hf (and GPTQ conversion) can start from it: config.json + model.safetensors, q / k rows permuted
to HF's rotate_half layout (the inverse of hf_init._unpermute). Checks the round trip: the export loaded back through
init_from_hf("fp32") gives the original model's loss on the first --check windows of --val.

  python scripts/analysis/export_hf_llama.py checkpoints/fp32_baseline/ckpt.pt OUT_DIR [--val data/wiki32k_val.bin]
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ttf.model import BitTransformer                                  # noqa: E402
from ttf.hf_init import init_from_hf, model_config_from_hf, _LIN     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--val", default="data/wiki32k_val.bin")
ap.add_argument("--check", type=int, default=4)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
assert ck["mode"] == "fp32", ck["mode"]
c, sd = ck["cfg"], ck["model"]


def permute(w, n_heads):
    """interleaved pairs -> HF rotate_half layout"""
    out, k = w.shape
    return w.view(n_heads, out // n_heads // 2, 2, k).transpose(1, 2).reshape(out, k)


hf = {"model.embed_tokens.weight": sd["tok_emb.weight"], "model.norm.weight": sd["norm.weight"]}
if "lm_head.weight" in sd:
    hf["lm_head.weight"] = sd["lm_head.weight"]
for i in range(c.n_layers):
    p = f"model.layers.{i}."
    hf[p + "input_layernorm.weight"] = sd[f"layers.{i}.attn_norm.weight"]
    hf[p + "post_attention_layernorm.weight"] = sd[f"layers.{i}.ffn_norm.weight"]
    for name_hf, (sub, name) in _LIN.items():
        W = sd[f"layers.{i}.{sub}.{name}.weight"]
        if name == "wq":
            W = permute(W, c.n_heads)
        elif name == "wk":
            W = permute(W, c.n_kv_heads)
        hf[p + name_hf + ".weight"] = W
os.makedirs(a.out, exist_ok=True)
from safetensors.torch import save_file                                   # noqa: E402
save_file({k: v.contiguous().float() for k, v in hf.items()}, os.path.join(a.out, "model.safetensors"))
json.dump({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": c.vocab_size,
           "hidden_size": c.dim, "num_hidden_layers": c.n_layers, "num_attention_heads": c.n_heads,
           "num_key_value_heads": c.n_kv_heads, "intermediate_size": c.hidden_dim, "rope_theta": c.rope_theta,
           "rms_norm_eps": c.norm_eps, "tie_word_embeddings": bool(c.tie_embeddings), "hidden_act": "silu",
           "max_position_embeddings": c.max_seq_len, "torch_dtype": "float32"},
          open(os.path.join(a.out, "config.json"), "w"), indent=1)

dev = "cuda"
mk = lambda: BitTransformer(c, grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev).eval()
m0 = mk(); m0.load_state_dict(sd, strict=True)
m1 = BitTransformer(model_config_from_hf(a.out, c.max_seq_len), grad_checkpoint=False,
                    make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev).eval()
init_from_hf(m1, a.out, "fp32", log=lambda *_: None)
val = np.memmap(a.val, dtype=np.uint16, mode="r")
with torch.no_grad():
    for i in range(a.check):
        x = torch.tensor(val[i * 2048:(i + 1) * 2048 + 1].astype(np.int64), device=dev)[None]
        l0, l1 = float(m0(x[:, :-1], x[:, 1:])[1]), float(m1(x[:, :-1], x[:, 1:])[1])
        print(f"window {i}: original {l0:.5f}  export {l1:.5f}")
        assert abs(l0 - l1) < 1e-3, "round trip mismatch"
print(f"exported to {a.out}")
