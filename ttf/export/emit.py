"""ttf.export.emit: model -> .ttf (GGUF) with the block-level IR emitted by the module library.

Each module class has an emitter (EMIT[class](ctx, module, path, inputs) -> outputs) that
writes its IR nodes. A model is exported by calling the emitter of its top module; no per-model code beyond the
emitters of the module classes it is made of. Tensors are written under their state-dict names verbatim, plus the
weight-derived constants (`<p>.beta`, `layers.i.attn.qk_scale`).
  python -m ttf.export.emit core CKPT.safetensors OUT.ttf [--tokenizer DIR]   (+ a reference-runner check)
  python -m ttf.export.emit core CKPT.safetensors OUT.ttf --float-check           (the float-path end-to-end test)
"""
import json, math, os, sys
import torch

from ..layers import KernelTernaryLinear
from .. import model as TM
from ..kernel import trit_nnz, beta_from_nnz
from .io import write_ttf, read_ttf
from . import ir as ttf_ir

EMIT = {}          # module class -> emitter
EMIT_BY_NAME = {}  # class name -> emitter (families whose module classes live outside ttf: registered without importing)
PART_TENSORS = ("wpacked", "row_scale", "col_scale", "in_mean", "out_bias", "mag_A", "mag_B")


def emits(cls):
    def reg(f):
        if isinstance(cls, str):
            EMIT_BY_NAME[cls] = f
        else:
            EMIT[cls] = f
        return f
    return reg


class Ctx:
    """the exporter's state: graphs (one current), declared state, derived tensors, and the state-dict tensors the
    emitted nodes use (the file carries those + the derived ones, nothing else)"""

    def __init__(self, sd=None):
        self.graphs, self.state, self.derived, self.replace, self.used, self.n = {}, [], {}, {}, set(), 0
        self.sd = sd
        self.nodes = []
        self.cfg = None

    def begin(self, name, inputs):
        """start graph `name`; inputs: [(name, shape, dtype)]"""
        self.nodes = []
        self.graphs[name] = {"inputs": [{"name": n, "shape": s, "dtype": d} for n, s, d in inputs], "nodes": self.nodes}
        return [n for n, _, _ in inputs]

    def end(self, **outputs):
        g = self.graphs[list(self.graphs)[-1]]
        g["outputs"] = [{"name": k, "of": v} for k, v in outputs.items()]

    def v(self, hint="v"):
        self.n += 1
        return f"{hint}{self.n}"

    def use(self, *names):
        for n in names:
            if self.sd is not None:
                assert n in self.sd or n in self.derived or n in self.replace, f"no tensor {n}"
            self.used.add(n)

    def use_part(self, p):
        """a TTF linear part's tensors (those present)"""
        for t in PART_TENSORS:
            if self.sd is None or f"{p}.{t}" in self.sd:
                self.used.add(f"{p}.{t}")

    def node(self, op, ins, p=None, attr=None, n_out=1, hint="v"):
        outs = [self.v(hint) for _ in range(n_out)]
        d = {"op": op, "in": ins, "out": outs}
        if p is not None:
            d["p"] = p
        if attr:
            d["attr"] = attr
        self.nodes.append(d)
        return outs[0] if n_out == 1 else outs

    def add_state(self, sid, kind, **kw):
        if not any(s["id"] == sid for s in self.state):
            self.state.append({"id": sid, "kind": kind, **kw})
        return sid

    def emit(self, mod, path, *ins, **kw):
        f = EMIT.get(type(mod)) or EMIT_BY_NAME.get(type(mod).__name__)
        if f is None:
            raise NotImplementedError(f"no IR emitter for {type(mod).__module__}.{type(mod).__name__} at '{path}'")
        return f(self, mod, path, *ins, **kw)

    def tensors(self):
        """{name: tensor} for the file: the used state-dict tensors (replaced where an emitter re-laid one out) + derived"""
        out = {}
        for n in sorted(self.used):
            if n in self.replace:
                out[n] = self.replace[n]
            elif n in self.derived:
                continue
            else:
                out[n] = self.sd[n]
        out.update(self.derived)
        return out


def _beta_cuda(m):
    """beta evaluated on CUDA, as trained: torch's CUDA rsqrt is not correctly rounded (1 ulp vs CPU)"""
    if not m.use_beta:
        return torch.ones(1)
    assert torch.cuda.is_available(), "beta must be computed on CUDA (the trainer's value); no CUDA device"
    return beta_from_nnz(trit_nnz(m.wpacked.cuda(), m.K), m.N).float().reshape(1).cpu()


@emits(KernelTernaryLinear)
def _e_lin(ctx, m, p, x, acc=None):
    ctx.derived[p + ".beta"] = _beta_cuda(m)
    ctx.use_part(p)
    attr = {"K": m.K, "N": m.N, "act_bits": m.act_bits, "use_beta": bool(m.use_beta)}
    if getattr(m, "mag_A", None) is not None:
        attr["mag_rank"] = int(m.mag_A.shape[1])
    return ctx.node("ttf.linear", [x] + ([acc] if acc is not None else []), p, attr)


@emits(torch.nn.Linear)
def _e_flin(ctx, m, p, x):
    """a float linear: matmul with the stored weight (+ bias)"""
    ctx.use(p + ".weight", *([p + ".bias"] if m.bias is not None else []))
    return ctx.node("matmul", [x], p, {"K": m.in_features, "N": m.out_features, "bias": m.bias is not None})


@emits(torch.nn.Embedding)
def _e_femb(ctx, m, p, ids):
    ctx.use(p + ".weight")
    return ctx.node("ttf.embed", [ids], p, {"kind": "float"})


def intk_tensors(ctx, m, p):
    """an IntKMatrix's tensors for the file: codes nib2 (bits <= 4) or one byte per code, lo / s f32"""
    c = m.codes.detach().to(torch.uint8)
    if m.bits <= 4:
        c = (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()
    ctx.replace[p + ".codes"] = c.cpu()
    ctx.replace[p + ".lo"] = m.lo.detach().float().cpu()
    ctx.replace[p + ".s"] = m.s.detach().float().cpu()
    ctx.use(p + ".codes", p + ".lo", p + ".s")
    return {"kind": "intk", "bits": m.bits, "g": m.g, "K": m.K, "V": m.N}


@emits(torch.nn.SiLU)
def _e_silu(ctx, m, p, x):
    return ctx.node("silu", [x])


@emits(torch.nn.Tanh)
def _e_tanh(ctx, m, p, x):
    return ctx.node("tanh", [x])


@emits(torch.nn.Identity)
def _e_id(ctx, m, p, x):
    return x


@emits(torch.nn.Sequential)
def _e_seq(ctx, m, p, x):
    for i, l in enumerate(m):
        x = ctx.emit(l, f"{p}.{i}" if p else str(i), x)
    return x


@emits(TM.RMSNorm)
def _e_norm(ctx, m, p, x):
    ctx.use(p + ".weight")
    return ctx.node("rms_norm", [x], p, {"eps": m.eps})


@emits(TM.Attention)
def _e_attn(ctx, m, p, x, pos, layer):
    H, Hkv, hd = m.n_heads, m.n_kv, m.head_dim
    q = ctx.emit(m.wq, p + ".wq", x)
    k = ctx.emit(m.wk, p + ".wk", x)
    v = ctx.emit(m.wv, p + ".wv", x)
    q = ctx.node("reshape", [q], attr={"shape": ["T", H, hd]})
    k = ctx.node("reshape", [k], attr={"shape": ["T", Hkv, hd]})
    v = ctx.node("reshape", [v], attr={"shape": ["T", Hkv, hd]})
    if getattr(m, "qk_logscale", None) is not None:      # learnable per-head temperature: q * exp(s_h)
        ctx.derived[p + ".qk_scale"] = m.qk_logscale.detach().float().exp().reshape(-1, 1).cpu()
        q = ctx.node("mul", [q], attr={"w": p + ".qk_scale"})      # [T, H, hd] * qk_scale [H, 1] (trailing-dim broadcast)
    q = ctx.node("rope", [q, pos], attr={"theta": ctx.cfg.rope_theta, "style": "interleaved"})
    k = ctx.node("rope", [k, pos], attr={"theta": ctx.cfg.rope_theta, "style": "interleaved"})
    sid = ctx.add_state(f"kv.{layer}", "kv", heads_kv=Hkv, head_dim=hd)
    o = ctx.node("attn", [q, k, v, pos], attr={"causal": True, "state": sid, "scale": 1.0 / math.sqrt(hd)})
    o = ctx.node("reshape", [o], attr={"shape": ["T", H * hd]})
    return ctx.emit(m.wo, p + ".wo", o)


@emits(TM.FeedForward)
def _e_ffn(ctx, m, p, x):
    g = ctx.emit(m.w_gate, p + ".w_gate", x)
    u = ctx.emit(m.w_up, p + ".w_up", x)
    a = ctx.node("glu", [g, u], attr={"act": "silu", "order": "gate|up"})
    return ctx.emit(m.w_down, p + ".w_down", a)


@emits(TM.Block)
def _e_block(ctx, m, p, x, pos, layer):
    h = ctx.emit(m.attn_norm, p + ".attn_norm", x)
    x = ctx.node("add", [x, ctx.emit(m.attn, p + ".attn", h, pos, layer)], hint="x")
    h = ctx.emit(m.ffn_norm, p + ".ffn_norm", x)
    return ctx.node("add", [x, ctx.emit(m.ffn, p + ".ffn", h)], hint="x")


@emits(TM.BitTransformer)
def _e_bit(ctx, m, p, ids, pos):
    ctx.cfg = m.c
    ctx.use("tok_emb.weight")
    x = ctx.node("ttf.embed", [ids], "tok_emb", {"kind": "float"}, hint="x")
    for i, l in enumerate(m.layers):
        x = ctx.emit(l, f"layers.{i}", x, pos, i)
    h = ctx.emit(m.norm, "norm", x)
    head = "tok_emb" if m.lm_head is None else "lm_head"
    ctx.use(head + ".weight")
    return ctx.node("ttf.head", [h], head, {"tied": m.lm_head is None, "out": "logits"}, hint="logits")


def build_ir(model, family, driver, sd=None):
    """the core LM: one graph step(ids, pos) -> logits. Returns (ir, derived) or, with sd, (ir, tensors for the file)"""
    ctx = Ctx(sd)
    ids, pos = ctx.begin("step", [("ids", ["T"], "i32"), ("pos", ["T"], "i32")])
    ctx.end(logits=ctx.emit(model, "", ids, pos))
    ir = {"ir_version": 1, "dims": {"T": "dynamic"}, "state": ctx.state, "graphs": ctx.graphs, "driver": driver}
    return ir, (ctx.tensors() if sd is not None else ctx.derived)


class float_path:
    """context: PyTorch's TTF products in float (fp64, no quantiser) and the runner likewise (IR I.9's float path)"""

    def __enter__(self):
        from .. import layers as TL
        self.TL = TL

        def apply(x, wpacked, layer):
            beta = layer._cached_beta(wpacked) if layer.use_beta else None
            y = torch.nn.functional.linear(x.double(), layer.w8().double())
            return (y * beta.double() if beta is not None else y).to(x.dtype)
        self.orig, self.fused = TL._KernelTernFn.apply, TL.FUSED_LAYER[0]
        TL._KernelTernFn.apply = staticmethod(apply)
        TL.FUSED_LAYER[0] = False                   # (the fused layer path does not go through _KernelTernFn)
        ttf_ir.FLOAT_PATH[0] = True
        return self

    def __exit__(self, *e):
        self.TL._KernelTernFn.apply = self.orig
        self.TL.FUSED_LAYER[0] = self.fused
        ttf_ir.FLOAT_PATH[0] = False


def load_core_safetensors(path):
    from safetensors import safe_open
    from ..config import ModelConfig
    from .. import layers as TL
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}
    cfg = json.loads(meta["ttf_config"])
    beta = bool(cfg.pop("beta", True))
    c = ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__})
    m = TL.build_kernel_transformer(c, grad_checkpoint=False, beta=beta)
    for l in m.modules():
        if isinstance(l, KernelTernaryLinear):
            l.enable_rc_scale()
    for n, l in m.named_modules():
        if isinstance(l, KernelTernaryLinear) and n + ".mag_A" in sd:
            l.enable_lowrank_mag("add", sd[n + ".mag_A"].shape[1])
        if isinstance(l, TM.Attention) and n + ".qk_logscale" in sd:
            l.qk_logscale = torch.nn.Parameter(torch.zeros(l.n_heads))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "freqs_cis" and not k.endswith(".frozen_scale")]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    for l in m.modules():
        if hasattr(l, "_invalidate"):
            l._invalidate()
    return m, sd, cfg, meta


def main():
    which, ck, out = sys.argv[1:4]
    tok = sys.argv[sys.argv.index("--tokenizer") + 1] if "--tokenizer" in sys.argv else None
    assert which == "core"
    m, sd, cfg, meta = load_core_safetensors(ck)
    m.eval()
    ir, tensors = build_ir(m, "bit_transformer", {"name": "lm", "graphs": {"step": "step"}, "attr": {}}, sd)
    ops = {}
    for n in ir["graphs"]["step"]["nodes"]:
        ops[n["op"]] = ops.get(n["op"], 0) + 1
    print("IR:", len(ir["graphs"]["step"]["nodes"]), "nodes", ops, "| state", len(ir["state"]), flush=True)
    write_ttf(out, tensors, {}, ir, "bit_transformer", cfg, "bf16", tok, meta)
    print("wrote", out, f"{os.path.getsize(out) / 2 ** 20:.1f} MiB", flush=True)

    # --- check: the reference runner on the FILE (prefill 12, then prefill 8 + 4 decode steps) vs PyTorch bf16
    kv, W = read_ttf(out)
    ir2 = json.loads(kv["ttf.ir"])
    D = ttf_ir.DTYPES[kv["ttf.compute_dtype"]]
    torch.manual_seed(0)
    ids = torch.randint(0, cfg["vocab_size"], (12,))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mm = m.to(dev).to(torch.bfloat16)
    mm.freqs_cis = TM.precompute_rope(mm.c.dim // mm.c.n_heads, mm.c.max_seq_len * 2, mm.c.rope_theta).to(dev)
    with torch.no_grad():
        ref = mm(ids[None].to(dev))[0][0].float().cpu()          # [12, V]
    R = ttf_ir.Runner(ir2, W, D)
    full = R("step", ids=ids, pos=torch.arange(12))["logits"]
    R2 = ttf_ir.Runner(ir2, W, D)
    a = R2("step", ids=ids[:8], pos=torch.arange(8))["logits"]
    steps = [R2("step", ids=ids[i:i + 1], pos=torch.tensor([i]))["logits"] for i in range(8, 12)]
    inc = torch.cat([a] + steps)
    rel = lambda x, y: float((x - y).norm() / y.norm())
    print(f"runner (file) vs PyTorch: rel L2 {rel(full, ref):.3e}, top-1 agree {float((full.argmax(-1) == ref.argmax(-1)).float().mean()):.3f}")
    print(f"prefill 8 + 4 decode vs prefill 12 (same runner code): rel L2 {rel(inc, full):.3e}, "
          f"top-1 agree {float((inc.argmax(-1) == full.argmax(-1)).float().mean()):.3f}")


def float_check(m, out, n=12, n_pre=8, dev="cuda"):
    """the float-path end-to-end test (IR I.9): PyTorch fp32 with the TTF product in float (no quantiser) vs the
    reference runner on the file, same mode; also prefill n_pre + single decode steps vs prefill n on the runner.
    Returns (rel L2 runner vs PyTorch, rel L2 incremental vs full, top-1 agreement)."""
    kv, W = read_ttf(out)
    ir2 = json.loads(kv["ttf.ir"])
    with float_path():
        mm = m.to(dev).float()
        mm.freqs_cis = TM.precompute_rope(mm.c.dim // mm.c.n_heads, mm.c.max_seq_len * 2, mm.c.rope_theta).to(dev)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, mm.c.vocab_size, (n,), generator=g)
        with torch.no_grad():
            ref = mm(ids[None].to(dev))[0][0].float().cpu()
        W = {k: (v.float() if v.is_floating_point() else v) for k, v in W.items()}
        full = ttf_ir.Runner(ir2, W, torch.float32)("step", ids=ids, pos=torch.arange(n))["logits"]
        R2 = ttf_ir.Runner(ir2, W, torch.float32)
        a = R2("step", ids=ids[:n_pre], pos=torch.arange(n_pre))["logits"]
        inc = torch.cat([a] + [R2("step", ids=ids[i:i + 1], pos=torch.tensor([i]))["logits"] for i in range(n_pre, n)])
    rel = lambda x, y: float((x - y).norm() / y.norm())
    return rel(full, ref), rel(inc, full), float((full.argmax(-1) == ref.argmax(-1)).float().mean())


if __name__ == "__main__":
    if "--float-check" in sys.argv:
        m, *_ = load_core_safetensors(sys.argv[2]); m.eval()
        r, i, t1 = float_check(m, sys.argv[3])
        print(f"FLOAT PATH fp32: runner (file) vs PyTorch rel L2 {r:.3e}; prefill 8 + 4 decode vs prefill 12 rel L2 "
              f"{i:.3e}; top-1 {t1:.3f}")
        sys.exit()
    main()
