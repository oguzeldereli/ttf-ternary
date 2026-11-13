"""ttf.export.qwen35: IR emitters for Qwen3.5 (HF transformers' qwen3_5 text model inside ttf.qwen.QwenTern): the
hybrid stack of gated full attention and Gated DeltaNet layers.

One graph `step(ids [T], pos [T]) -> logits [T, V]` for the lm driver; state per layer: kv.i (attention layers),
conv.i + rec.i (DeltaNet layers). Text-only positions: the three M-RoPE axes carry the same position, so the rotation
is the plain partial NEOX one (`style half`, `dims` = head_dim x partial_rotary_factor).
  python -m ttf.export.qwen35 HF_DIR CKPT OUT.ttf [--check]   (a ttfconv-converted or ttf-trained Qwen3.5 checkpoint)
"""
import json
import math
import os
import sys

import torch

from .emit import Ctx, emits, intk_tensors, float_path
from .io import write_ttf, read_ttf
from . import ir as ttf_ir


@emits("CenteredLinear")
def _e_clin(ctx, m, p, x):
    """F.linear(x - mu, W) + bias (ttf.qwen.CenteredLinear)"""
    ctx.use(p + ".weight", p + ".in_mean", p + ".out_bias")
    x = ctx.node("sub", [x], attr={"w": p + ".in_mean"})
    y = ctx.node("matmul", [x], p, {"K": m.in_features, "N": m.out_features, "bias": False})
    return ctx.node("add", [y], attr={"w": p + ".out_bias"})


@emits("_Emb")
def _e_intk_emb(ctx, m, p, ids):
    return ctx.node("ttf.embed", [ids], p + ".m", intk_tensors(ctx, m.m, p + ".m"))


@emits("_Head")
def _e_intk_head(ctx, m, p, h):
    return ctx.node("ttf.head", [h], p + ".m", {**intk_tensors(ctx, m.m, p + ".m"), "out": "logits"})


@emits("Qwen3_5RMSNorm")
def _e_qnorm(ctx, m, p, x):
    ctx.use(p + ".weight")
    return ctx.node("rms_norm", [x], p, {"eps": m.eps, "offset": True})       # x * (1 + w)


@emits("Qwen3_5MLP")
def _e_qmlp(ctx, m, p, x):
    assert m.config.hidden_act == "silu"
    g = ctx.emit(m.gate_proj, p + ".gate_proj", x)
    u = ctx.emit(m.up_proj, p + ".up_proj", x)
    return ctx.emit(m.down_proj, p + ".down_proj", ctx.node("glu", [g, u], attr={"act": "silu", "order": "gate|up"}))


@emits("Qwen3_5Attention")
def _e_qattn(ctx, m, p, x, pos, layer):
    """gated attention: q_proj gives (q, gate) per head; q / k RMS-normed per head; out = o_proj(attn * sigmoid(gate))"""
    c = m.config
    H, Hkv, hd = c.num_attention_heads, c.num_key_value_heads, m.head_dim
    qg = ctx.node("reshape", [ctx.emit(m.q_proj, p + ".q_proj", x)], attr={"shape": ["T", H, 2 * hd]})
    q = ctx.node("slice", [qg], attr={"dim": -1, "start": 0, "len": hd})
    gate = ctx.node("reshape", [ctx.node("slice", [qg], attr={"dim": -1, "start": hd, "len": hd})],
                    attr={"shape": ["T", H * hd]})
    q = ctx.emit(m.q_norm, p + ".q_norm", q)
    k = ctx.node("reshape", [ctx.emit(m.k_proj, p + ".k_proj", x)], attr={"shape": ["T", Hkv, hd]})
    k = ctx.emit(m.k_norm, p + ".k_norm", k)
    v = ctx.node("reshape", [ctx.emit(m.v_proj, p + ".v_proj", x)], attr={"shape": ["T", Hkv, hd]})
    q = ctx.node("rope", [q, pos], attr=ctx.rope)
    k = ctx.node("rope", [k, pos], attr=ctx.rope)
    sid = ctx.add_state(f"kv.{layer}", "kv", heads_kv=Hkv, head_dim=hd)
    o = ctx.node("attn", [q, k, v, pos], attr={"causal": True, "state": sid, "scale": m.scaling})
    o = ctx.node("reshape", [o], attr={"shape": ["T", H * hd]})
    o = ctx.node("mul", [o, ctx.node("sigmoid", [gate])])
    return ctx.emit(m.o_proj, p + ".o_proj", o)


@emits("Qwen3_5GatedDeltaNet")
def _e_qdelta(ctx, m, p, x, pos, layer):
    """in_proj_qkv -> causal depthwise conv (k 4, silu; state conv.i) -> q / k / v; beta = sigmoid(in_proj_b x);
    g = -exp(A_log) softplus(in_proj_a x + dt_bias); delta rule (state rec.i); gated RMS norm with silu(in_proj_z x);
    out_proj"""
    assert m.activation == "silu" and m.conv1d.bias is None
    Hk, Hv, dk, dv = m.num_k_heads, m.num_v_heads, m.head_k_dim, m.head_v_dim
    C, kw = m.conv_dim, m.conv_kernel_size
    qkv = ctx.emit(m.in_proj_qkv, p + ".in_proj_qkv", x)                                       # [T, C]
    ctx.use(p + ".conv1d.weight")
    cs = ctx.add_state(f"conv.{layer}", "conv", channels=C, len=kw - 1)
    qkv = ctx.node("conv1d_causal", [ctx.node("permute", [qkv], attr={"dims": [1, 0]})], p + ".conv1d",
                   {"ctx": kw - 1, "stride": 1, "dilation": 1, "groups": C, "kernel": kw, "bias": False, "act": "silu",
                    "state": cs})
    qkv = ctx.node("permute", [qkv], attr={"dims": [1, 0]})                                   # [T, C]
    q = ctx.node("reshape", [ctx.node("slice", [qkv], attr={"dim": -1, "start": 0, "len": m.key_dim})],
                 attr={"shape": ["T", Hk, dk]})
    k = ctx.node("reshape", [ctx.node("slice", [qkv], attr={"dim": -1, "start": m.key_dim, "len": m.key_dim})],
                 attr={"shape": ["T", Hk, dk]})
    v = ctx.node("reshape", [ctx.node("slice", [qkv], attr={"dim": -1, "start": 2 * m.key_dim, "len": m.value_dim})],
                 attr={"shape": ["T", Hv, dv]})
    z = ctx.node("reshape", [ctx.emit(m.in_proj_z, p + ".in_proj_z", x)], attr={"shape": ["T", Hv, dv]})
    b = ctx.node("sigmoid", [ctx.emit(m.in_proj_b, p + ".in_proj_b", x)])
    ctx.use(p + ".dt_bias")
    ctx.derived[p + ".neg_exp_A"] = (-m.A_log.detach().float().exp()).cpu()                  # -exp(A_log), fp32
    a = ctx.node("add", [ctx.emit(m.in_proj_a, p + ".in_proj_a", x)], attr={"w": p + ".dt_bias"})
    g = ctx.node("mul", [ctx.node("softplus", [a])], attr={"w": p + ".neg_exp_A"})
    rs = ctx.add_state(f"rec.{layer}", "rec", heads=Hv, hd_k=dk, hd_v=dv)
    o = ctx.node("delta_rule", [q, k, v, g, b], attr={"state": rs, "qk_l2norm": True, "scale": dk ** -0.5,
                                                      "heads_k": Hk, "heads_v": Hv})
    ctx.use(p + ".norm.weight")
    o = ctx.node("rms_norm", [o], p + ".norm", {"eps": m.norm.variance_epsilon})            # norm before the gate
    o = ctx.node("mul", [o, ctx.node("silu", [z])])
    o = ctx.node("reshape", [o], attr={"shape": ["T", Hv * dv]})
    return ctx.emit(m.out_proj, p + ".out_proj", o)


@emits("Qwen3_5DecoderLayer")
def _e_qlayer(ctx, m, p, x, pos, layer):
    h = ctx.emit(m.input_layernorm, p + ".input_layernorm", x)
    if m.block_type == "linear_attention":
        h = ctx.emit(m.linear_attn, p + ".linear_attn", h, pos, layer)
    else:
        h = ctx.emit(m.self_attn, p + ".self_attn", h, pos, layer)
    x = ctx.node("add", [x, h], hint="x")
    h = ctx.emit(m.post_attention_layernorm, p + ".post_attention_layernorm", x)
    return ctx.node("add", [x, ctx.emit(m.mlp, p + ".mlp", h)], hint="x")


def _rope(tm):
    r = tm.rotary_emb
    assert r.rope_type == "default" and float(r.attention_scaling) == 1.0, (r.rope_type, r.attention_scaling)
    rp = tm.config.rope_parameters
    return {"theta": float(rp["rope_theta"]), "style": "half", "dims": 2 * r.inv_freq.numel()}


@emits("QwenTern")
def _e_qwentern(ctx, m, p, ids, pos):
    tm = m.hf.model
    ctx.rope = _rope(tm)
    x = ctx.emit(tm.embed_tokens, "hf.model.embed_tokens", ids)
    for i, l in enumerate(tm.layers):
        x = ctx.emit(l, f"hf.model.layers.{i}", x, pos, i)
    h = ctx.emit(tm.norm, "hf.model.norm", x)
    lm = m.hf.lm_head
    if isinstance(lm, torch.nn.Linear) and lm.weight is getattr(tm.embed_tokens, "weight", None):
        ctx.use("hf.model.embed_tokens.weight")                                                  # tied
        return ctx.node("ttf.head", [h], "hf.model.embed_tokens", {"tied": True, "out": "logits"}, hint="logits")
    if isinstance(lm, torch.nn.Linear):
        assert lm.bias is None
        ctx.use("hf.lm_head.weight")
        return ctx.node("ttf.head", [h], "hf.lm_head", {"tied": False, "out": "logits"}, hint="logits")
    return ctx.emit(lm, "hf.lm_head", h)


def build(m, sd, eos=None):
    ctx = Ctx(sd)
    ids, pos = ctx.begin("step", [("ids", ["T"], "i32"), ("pos", ["T"], "i32")])
    ctx.end(logits=ctx.emit(m, "", ids, pos))
    ir = {"ir_version": 1, "dims": {"T": "dynamic"}, "state": ctx.state, "graphs": ctx.graphs,
          "driver": {"name": "lm", "graphs": {"step": "step"}, "attr": {"eos": eos or []}}}
    return ir, ctx


def export(m, hf_dir, out, meta=None):
    sd = m.state_dict()
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    eos = tc.get("eos_token_id")
    ir, ctx = build(m, sd, eos if isinstance(eos, list) else ([eos] if eos is not None else []))
    D = {torch.bfloat16: "bf16", torch.float32: "f32"}[m.hf.model.norm.weight.dtype]
    write_ttf(out, ctx.tensors(), {}, ir, "qwen3_5", cfg, D, hf_dir, meta)
    return ir


def _ref_delta(query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
               **kw):
    """HF's recurrent gated delta rule in fp32, plain torch (no FLA kernel: its chunked Triton path uses TF32 dots)"""
    dt = query.dtype
    q, k, v, b, d = (x.transpose(1, 2).float() for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q / (q.shape[-1] ** 0.5)
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1], device=v.device) if initial_state is None else initial_state.float()
    o = torch.zeros_like(v)
    for t in range(T):
        S = S * d[:, :, t].exp()[..., None, None]
        dl = (v[:, :, t] - (S * k[:, :, t, :, None]).sum(-2)) * b[:, :, t, None]
        S = S + k[:, :, t, :, None] * dl[:, :, None, :]
        o[:, :, t] = (S * q[:, :, t, :, None]).sum(-2)
    return o.transpose(1, 2).contiguous().to(dt), (S if output_final_state else None)


class _ref_kernels:
    """PyTorch side of the check on the plain-torch delta rule"""
    def __enter__(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
        self.Q, self.o = Q, (Q.torch_chunk_gated_delta_rule, Q.torch_recurrent_gated_delta_rule)
        Q.torch_chunk_gated_delta_rule = Q.torch_recurrent_gated_delta_rule = _ref_delta
        return self

    def __exit__(self, *e):
        self.Q.torch_chunk_gated_delta_rule, self.Q.torch_recurrent_gated_delta_rule = self.o


@torch.no_grad()
def check(m, path, n=12, n_pre=8, dev="cuda", float_mode=True):
    """the runner on the file vs PyTorch (no cache, full prefill), and prefill n_pre + single decode steps vs prefill n
    on the runner (kv, conv and delta-rule state)"""
    import contextlib
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kv, W = read_ttf(path)
    ir = json.loads(kv["ttf.ir"])
    D = ttf_ir.DTYPES[kv["ttf.compute_dtype"]]
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 50000, (n,), generator=g)
    with (float_path() if float_mode else contextlib.nullcontext()):
        with _ref_kernels():
            ref = m(ids[None].to(dev))[0][0].float().cpu()
        R = ttf_ir.Runner(ir, W, D, dev)
        full = R("step", ids=ids, pos=torch.arange(n))["logits"].cpu()
        R.reset()
        a = R("step", ids=ids[:n_pre], pos=torch.arange(n_pre))["logits"].cpu()
        inc = torch.cat([a] + [R("step", ids=ids[i:i + 1], pos=torch.tensor([i]))["logits"].cpu() for i in range(n_pre, n)])
    rel = lambda x, y: float((x - y).norm() / y.norm())
    return {"runner vs PyTorch": rel(full, ref), f"prefill {n_pre} + {n - n_pre} decode vs prefill {n}": rel(inc, full),
            "top-1 agree": float((full.argmax(-1) == ref.argmax(-1)).float().mean())}


def check_slice(m, hf_dir, out, n, dev="cuda", dtype=None):
    """a model too big for one GPU (27B): keep its first n decoder layers (embedding, final norm and head as they are),
    move that to the GPU, export it to `out` and run check on it. Checks the emitters on this model's own modules and
    tensors (n >= full_attention_interval reaches an attention layer)."""
    tm = m.hf.model
    tm.layers = torch.nn.ModuleList(list(tm.layers)[:n])
    tm.config.num_hidden_layers = n
    tm.config.layer_types = tm.config.layer_types[:n]
    m.to(dev)
    if dtype is not None:
        m.to(dtype)
    export(m, hf_dir, out, {"slice": n})
    return check(m, out, dev=dev)


if __name__ == "__main__":
    hf_dir, ck, out = sys.argv[1:4]
    from ttfconv.model import load_converted
    dev = "cpu" if "--cpu" in sys.argv else "cuda"             # --cpu: a model larger than the GPU (27B); beta on CUDA
    m, _ = load_converted(hf_dir, ck, dev)
    dt = torch.float32 if "--f32" in sys.argv else (torch.bfloat16 if "--bf16" in sys.argv else None)
    m = m.eval()
    if "--check-slice" in sys.argv:                             # (the cast after the cut: the slice only)
        n = int(sys.argv[sys.argv.index("--check-slice") + 1])
        print(f"slice {n} layers, float path", check_slice(m, hf_dir, out, n, dtype=dt), flush=True)
        sys.exit(0)
    if dt is not None:                                          # else the checkpoint's own dtypes
        m = m.to(dt)
    ir = export(m, hf_dir, out, {"ckpt": os.path.abspath(ck)})
    ops = {}
    for nd in ir["graphs"]["step"]["nodes"]:
        ops[nd["op"]] = ops.get(nd["op"], 0) + 1
    print("IR:", len(ir["graphs"]["step"]["nodes"]), "nodes", ops, "| state", len(ir["state"]), flush=True)
    print("wrote", out, f"{os.path.getsize(out) / 2 ** 20:.1f} MiB", flush=True)
    if "--check" in sys.argv:
        for fm in ((True, False) if "--int8" in sys.argv else (True,)):
            print("float path" if fm else "int8 path", check(m, out, float_mode=fm), flush=True)
