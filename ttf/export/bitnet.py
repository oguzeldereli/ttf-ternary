"""ttf.export.bitnet: Microsoft's BitNet b1.58 (HF model_type "bitnet", e.g. microsoft/bitnet-b1.58-2B-4T) into a .ttf,
straight from config.json + model.safetensors (no HF quantizer needed).

BitNet's linear (HF AutoBitLinear, offline mode): y = linear(fakequant(x), W) * weight_scale, with
  W ternary, packed 4 per byte along the output rows (rows [i N/4, (i+1) N/4) in bits 2i..2i+1, value - 1),
  fakequant per token: s = 127 / max|x| (>= 1e-5), round(x s) clamped to [-128, 127], / s;
  weight_scale one bf16 scalar per matrix.
That is our ttf.linear with beta = weight_scale and no centring / row / column scales (our reference rounds x / (max/127);
the product is the exact int8 x trit sum, as bitnet.cpp, where HF's own forward multiplies in bf16).
The block: Llama attention with attn_sub_norm before o_proj, MLP down(ffn_sub_norm(relu(gate)^2 * up)), tied head.
  python -m ttf.export.bitnet HF_DIR OUT.ttf [--check]
"""
import json
import os
import sys

import torch

from .emit import Ctx
from .io import write_ttf, read_ttf
from . import ir as ttf_ir
from ..kernel import pack_rows


def unpack_bitnet(packed, N):
    """uint8 [N/4, K] -> int8 [N, K] trits"""
    rows = packed.shape[0]
    assert rows * 4 == N, (packed.shape, N)
    return torch.cat([((packed >> (2 * i)) & 3).to(torch.int8) - 1 for i in range(4)])


def build(cfg, sd):
    """the IR and the file's tensors from HF's config and state dict"""
    ctx = Ctx(sd)
    d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    H, Hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, F_, eps = d // H, cfg["intermediate_size"], cfg["rms_norm_eps"]
    assert cfg["hidden_act"] == "relu2" and cfg.get("tie_word_embeddings", True)
    rope = {"theta": float(cfg["rope_theta"]), "style": "half", "dims": hd}

    def lin(p, x, N, K):
        q = sd[p + ".weight"]
        if q.dtype == torch.uint8:                                    # packed (offline checkpoint)
            t = unpack_bitnet(q, N)
        else:                                                         # bf16 trits (an unpacked copy)
            t = q.round().to(torch.int8)
        assert t.shape == (N, K) and int(t.abs().max()) <= 1
        ctx.replace[p + ".wpacked"] = pack_rows(t)
        ctx.derived[p + ".beta"] = sd[p + ".weight_scale"].float().reshape(1)
        ctx.use(p + ".wpacked")
        return ctx.node("ttf.linear", [x], p, {"K": K, "N": N, "act_bits": 8, "use_beta": True})

    def norm(p, x, size=None):
        ctx.use(p + ".weight")
        return ctx.node("rms_norm", [x], p, {"eps": eps})

    ids, pos = ctx.begin("step", [("ids", ["T"], "i32"), ("pos", ["T"], "i32")])
    ctx.use("model.embed_tokens.weight")
    x = ctx.node("ttf.embed", [ids], "model.embed_tokens", {"kind": "float"}, hint="x")
    for i in range(L):
        p = f"model.layers.{i}"
        h = norm(p + ".input_layernorm", x)
        a = p + ".self_attn"
        q = ctx.node("reshape", [lin(a + ".q_proj", h, H * hd, d)], attr={"shape": ["T", H, hd]})
        k = ctx.node("reshape", [lin(a + ".k_proj", h, Hkv * hd, d)], attr={"shape": ["T", Hkv, hd]})
        v = ctx.node("reshape", [lin(a + ".v_proj", h, Hkv * hd, d)], attr={"shape": ["T", Hkv, hd]})
        q, k = ctx.node("rope", [q, pos], attr=rope), ctx.node("rope", [k, pos], attr=rope)
        sid = ctx.add_state(f"kv.{i}", "kv", heads_kv=Hkv, head_dim=hd)
        o = ctx.node("attn", [q, k, v, pos], attr={"causal": True, "state": sid, "scale": hd ** -0.5})
        o = norm(a + ".attn_sub_norm", ctx.node("reshape", [o], attr={"shape": ["T", H * hd]}))
        x = ctx.node("add", [x, lin(a + ".o_proj", o, d, H * hd)], hint="x")
        h = norm(p + ".post_attention_layernorm", x)
        mp = p + ".mlp"
        g = ctx.node("glu", [lin(mp + ".gate_proj", h, F_, d), lin(mp + ".up_proj", h, F_, d)],
                     attr={"act": "relu2", "order": "gate|up"})
        x = ctx.node("add", [x, lin(mp + ".down_proj", norm(mp + ".ffn_sub_norm", g), d, F_)], hint="x")
    h = norm("model.norm", x)
    ctx.end(logits=ctx.node("ttf.head", [h], "model.embed_tokens", {"tied": True, "out": "logits"}, hint="logits"))
    eos = cfg.get("eos_token_id")
    ir = {"ir_version": 1, "dims": {"T": "dynamic"}, "state": ctx.state, "graphs": ctx.graphs,
          "driver": {"name": "lm", "graphs": {"step": "step"},
                     "attr": {"eos": eos if isinstance(eos, list) else ([eos] if eos is not None else [])}}}
    return ir, ctx


def export(hf_dir, out):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    assert cfg["model_type"] == "bitnet", cfg["model_type"]
    sd = load_file(os.path.join(hf_dir, "model.safetensors"))
    ir, ctx = build(cfg, sd)
    write_ttf(out, ctx.tensors(), {}, ir, "bitnet", cfg, "bf16", hf_dir, {"source": "hf:" + os.path.basename(hf_dir)})
    return ir


@torch.no_grad()
def check(hf_dir, path, n=24, dev="cuda"):
    """the runner on the file vs HF's BitNetForCausalLM (its bf16 fake-quant forward), on a real text prefix, and the
    runner's prefill + decode vs its prefill"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(hf_dir)
    text = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after "
            "the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.")
    ids = torch.tensor(tok(text)["input_ids"][:n])
    m = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.bfloat16).to(dev).eval()
    ref = m(ids[None].to(dev)).logits[0].float().cpu()
    del m
    torch.cuda.empty_cache()
    kv, W = read_ttf(path)
    R = ttf_ir.Runner(json.loads(kv["ttf.ir"]), W, ttf_ir.DTYPES[kv["ttf.compute_dtype"]], dev)
    full = R("step", ids=ids, pos=torch.arange(len(ids)))["logits"].cpu()
    R.reset()
    k = len(ids) - 4
    a = R("step", ids=ids[:k], pos=torch.arange(k))["logits"].cpu()
    inc = torch.cat([a] + [R("step", ids=ids[i:i + 1], pos=torch.tensor([i]))["logits"].cpu()
                           for i in range(k, len(ids))])
    rel = lambda x, y: float((x - y).norm() / y.norm())
    nll = lambda lg: float(torch.nn.functional.cross_entropy(lg[:-1], ids[1:]))
    return {"runner vs HF (rel L2)": rel(full, ref), "top-1 agree": float((full.argmax(-1) == ref.argmax(-1)).float().mean()),
            "nll runner": nll(full), "nll HF": nll(ref), f"prefill {k} + 4 decode vs prefill": rel(inc, full)}


if __name__ == "__main__":
    hf_dir, out = sys.argv[1:3]
    ir = export(hf_dir, out)
    ops = {}
    for nd in ir["graphs"]["step"]["nodes"]:
        ops[nd["op"]] = ops.get(nd["op"], 0) + 1
    print("IR:", len(ir["graphs"]["step"]["nodes"]), "nodes", ops, flush=True)
    print("wrote", out, f"{os.path.getsize(out) / 2 ** 20:.1f} MiB", flush=True)
    if "--check" in sys.argv:
        print(check(hf_dir, out), flush=True)
