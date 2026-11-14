"""ttf.export.bitnet: Microsoft's BitNet b1.58 (HF model_type "bitnet", e.g. microsoft/bitnet-b1.58-2B-4T) into a .ttf,
straight from config.json + model.safetensors (no HF quantizer needed).

BitNet's linear (HF AutoBitLinear, offline mode): y = linear(fakequant(x), W) * weight_scale, with
  W ternary, packed 4 per byte along the output rows (rows [i N/4, (i+1) N/4) in bits 2i..2i+1, value - 1),
  fakequant per token: s = 127 / max|x| (>= 1e-5), round(x s) clamped to [-128, 127], / s;
  weight_scale one bf16 scalar per matrix.
That is our ttf.linear with beta = weight_scale and no centring / row / column scales (our reference rounds x / (max/127);
the product is the exact int8 x trit sum, as bitnet.cpp, where HF's own forward multiplies in bf16).
The block: Llama attention with attn_sub_norm before o_proj, MLP down(ffn_sub_norm(relu(gate)^2 * up)), tied head.
Also the Llama-architecture 1.58-bit checkpoints (HF model_type "llama" with quantization_config quant_method "bitnet",
e.g. tiiuae/Falcon3-*-1.58bit): the same packing, HF's BitLinear y = linear(q(x), W) / (s_x weight_scale),
so beta = 1 / weight_scale; a plain Llama block (no sub-norms, SiLU GLU) and an untied bf16 lm_head.
--intk B (B <= 4): the float embedding and head stored as int-k codes, B bits per weight in groups of 64 (round to nearest
from each group's min / max: lossy, as llama.cpp's quantized token_embd / output), which ttf-home keeps packed; the
ternary layers are exact either way.
  python -m ttf.export.bitnet HF_DIR OUT.ttf [--check] [--intk B]
"""
import json
import os
import sys

import torch

from .emit import Ctx, intk_tensors
from .io import write_ttf, read_ttf
from . import ir as ttf_ir
from ..kernel import pack_rows


def unpack_bitnet(packed, N):
    """uint8 [N/4, K] -> int8 [N, K] trits"""
    rows = packed.shape[0]
    assert rows * 4 == N, (packed.shape, N)
    return torch.cat([((packed >> (2 * i)) & 3).to(torch.int8) - 1 for i in range(4)])


def kind(cfg):
    """"bitnet" (BitNet b1.58: sub-norms, relu2, tied head, out * weight_scale) or "llama" (a Llama checkpoint with
    BitLinear layers: out / weight_scale)"""
    if cfg["model_type"] == "bitnet":
        return "bitnet"
    q = cfg.get("quantization_config") or {}
    if cfg["model_type"] == "llama" and q.get("quant_method") == "bitnet":
        assert q.get("linear_class", "bitlinear") == "bitlinear" and not q.get("use_rms_norm"), q
        return "llama"
    raise ValueError(f"not a BitNet checkpoint: model_type {cfg['model_type']}, quantization_config {q}")


def _intk(ctx, p, bits):
    from ..intk_emb import IntKMatrix
    W = ctx.sd[p + ".weight"] if hasattr(ctx, "sd") else None
    m = IntKMatrix(W.cuda() if torch.cuda.is_available() else W, bits, 64)
    m.codes, m.lo.data, m.s.data = m.codes.cpu(), m.lo.data.cpu(), m.s.data.cpu()
    return intk_tensors(ctx, m, p)


def build(cfg, sd, intk=0):
    """the IR and the file's tensors from HF's config and state dict (intk: bits of the int-k embedding / head, 0 float)"""
    ctx = Ctx(sd)
    kd = kind(cfg)
    d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    H, Hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, F_, eps = cfg.get("head_dim") or d // H, cfg["intermediate_size"], cfg["rms_norm_eps"]
    act = {"bitnet": "relu2", "llama": "silu"}[kd]
    assert cfg["hidden_act"] == act, (cfg["hidden_act"], kd)
    tied = cfg.get("tie_word_embeddings", kd == "bitnet")
    assert kd == "llama" or tied
    assert not cfg.get("rope_scaling"), cfg.get("rope_scaling")
    assert not cfg.get("attention_bias") and not cfg.get("mlp_bias")
    rope = {"theta": float(cfg["rope_theta"]), "style": "half", "dims": hd}

    def lin(p, x, N, K):
        q = sd[p + ".weight"]
        if q.dtype == torch.uint8:                                    # packed (offline checkpoint)
            t = unpack_bitnet(q, N)
        else:                                                         # bf16 trits (an unpacked copy)
            t = q.round().to(torch.int8)
        assert t.shape == (N, K) and int(t.abs().max()) <= 1
        ctx.replace[p + ".wpacked"] = pack_rows(t)
        ws = sd[p + ".weight_scale"].float().reshape(1)
        ctx.derived[p + ".beta"] = ws if kd == "bitnet" else 1.0 / ws
        ctx.use(p + ".wpacked")
        return ctx.node("ttf.linear", [x], p, {"K": K, "N": N, "act_bits": 8, "use_beta": True})

    def norm(p, x, size=None):
        ctx.use(p + ".weight")
        return ctx.node("rms_norm", [x], p, {"eps": eps})

    ids, pos = ctx.begin("step", [("ids", ["T"], "i32"), ("pos", ["T"], "i32")])
    if intk:
        ea = _intk(ctx, "model.embed_tokens", intk)
        x = ctx.node("ttf.embed", [ids], "model.embed_tokens", ea, hint="x")
    else:
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
        o = ctx.node("reshape", [o], attr={"shape": ["T", H * hd]})
        if kd == "bitnet":
            o = norm(a + ".attn_sub_norm", o)
        x = ctx.node("add", [x, lin(a + ".o_proj", o, d, H * hd)], hint="x")
        h = norm(p + ".post_attention_layernorm", x)
        mp = p + ".mlp"
        g = ctx.node("glu", [lin(mp + ".gate_proj", h, F_, d), lin(mp + ".up_proj", h, F_, d)],
                     attr={"act": act, "order": "gate|up"})
        if kd == "bitnet":
            g = norm(mp + ".ffn_sub_norm", g)
        x = ctx.node("add", [x, lin(mp + ".down_proj", g, d, F_)], hint="x")
    h = norm("model.norm", x)
    if tied:
        hd_ = ctx.node("ttf.head", [h], "model.embed_tokens", {**(ea if intk else {}), "tied": True, "out": "logits"},
                       hint="logits")
    elif intk:
        hd_ = ctx.node("ttf.head", [h], "lm_head", {**_intk(ctx, "lm_head", intk), "tied": False, "out": "logits"},
                       hint="logits")
    else:
        ctx.use("lm_head.weight")
        hd_ = ctx.node("ttf.head", [h], "lm_head", {"tied": False, "out": "logits"}, hint="logits")
    ctx.end(logits=hd_)
    eos = cfg.get("eos_token_id")
    ir = {"ir_version": 1, "dims": {"T": "dynamic"}, "state": ctx.state, "graphs": ctx.graphs,
          "driver": {"name": "lm", "graphs": {"step": "step"},
                     "attr": {"eos": eos if isinstance(eos, list) else ([eos] if eos is not None else [])}}}
    return ir, ctx


def export(hf_dir, out, intk=0):
    from safetensors.torch import load_file
    import glob
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    kd = kind(cfg)
    sd = {}
    for f in sorted(glob.glob(os.path.join(hf_dir, "*.safetensors"))):     # (one file or shards)
        sd.update(load_file(f))
    ir, ctx = build(cfg, sd, intk)
    write_ttf(out, ctx.tensors(), {}, ir, "bitnet" if kd == "bitnet" else "llama-bitnet", cfg, "bf16", hf_dir,
              {"source": "hf:" + os.path.basename(hf_dir)})
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
    ir = export(hf_dir, out, int(sys.argv[sys.argv.index("--intk") + 1]) if "--intk" in sys.argv else 0)
    ops = {}
    for nd in ir["graphs"]["step"]["nodes"]:
        ops[nd["op"]] = ops.get(nd["op"], 0) + 1
    print("IR:", len(ir["graphs"]["step"]["nodes"]), "nodes", ops, flush=True)
    print("wrote", out, f"{os.path.getsize(out) / 2 ** 20:.1f} MiB", flush=True)
    if "--check" in sys.argv:
        print(check(hf_dir, out), flush=True)
