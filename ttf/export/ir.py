"""ttf.export.ir: the IR's Python reference.

Every IR op as plain torch: the semantic definition, and a reference runner (`Runner`) that executes a graph from a
.ttf file's IR + tensors with explicit state (KV cache per `kv` id, conv tails per `conv` id) and explicit positions,
prefill and decode with the same graph. It is what the C++ runtime is tested against and what proves an exporter's IR
before the runtime exists. Activations are f32 except where the Arithmetic asks for D (TTF parts: input rounded to D
at node entry). Leading dims are batch dims wherever an op allows them (x [..., K] for linears, [..., S, H, hd] for
bidirectional attention). Conv ops work on [C, T] (channels first, as PyTorch, batch 1).
"""
import math

import torch
import torch.nn.functional as F

FLOAT_PATH = [False]       # debug: TTF products in float, no quantiser (the float-path end-to-end test)
DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32, "f16": torch.float16}


def unpack_rows5(packed, K):
    """rows5 layout: uint8 [N, ceil(K/5)] base-3, digit j of byte i = trit 5 i + j (+1) -> int8 [N, K]"""
    p = packed.to(torch.int16)
    digs = []
    for _ in range(5):
        digs.append(p % 3)
        p = p // 3
    t = torch.stack(digs, -1).reshape(packed.shape[0], -1)[:, :K]
    return (t - 1).to(torch.int8)


def unpack_nib2(codes, K):
    """nib2 layout: uint8 [V, K/2], low nibble = even column -> uint8 [V, K]"""
    c = codes.to(torch.int16)
    return torch.stack([c & 15, c >> 4], -1).reshape(codes.shape[0], -1)[:, :K].to(torch.uint8)


def _mm_exact(q, T):
    """integer product, exact: fp32 when every partial sum stays below 2^24 (|q| <= 128), else fp64"""
    if 128 * T.shape[1] < 2 ** 24 and q.is_cuda:
        return q.float() @ T.float().t()
    return (q.double() @ T.double().t()).float()


def ttf_linear_part(x, W, p, attr, D):
    """Arithmetic steps 1-8 on one KernelTernaryLinear part; x [M, K] -> [M, N] in D"""
    g = lambda n: W.get(f"{p}.{n}")
    x = x.to(D)                                                      # node entry: rounded to D
    if g("in_mean") is not None:
        x = x - g("in_mean").to(D)
    if g("col_scale") is not None:
        x = x * g("col_scale").to(D)
    T = W[f"{p}.trits_i8"]                                           # unpacked once at load
    if FLOAT_PATH[0]:
        y = (x.double() @ T.double().t() * g("beta").double()).to(D)
        return _post(y, x, g, D)
    qp = 2 ** (attr.get("act_bits", 8) - 1) - 1
    amax = x.abs().amax(-1, keepdim=True)
    a = torch.maximum(amax, torch.tensor(1e-5, dtype=D, device=x.device)).float()
    s = a * torch.tensor(1.0 / qp, dtype=torch.float32)
    q = torch.round(x.float() / s).clamp_(-qp - 1, qp)
    acc = _mm_exact(q, T)
    t = ((acc * s) * g("beta").float()).to(torch.bfloat16)
    return _post(t.to(D), x, g, D)


def _post(y, x, g, D):
    if g("mag_A") is not None:                                       # step 6: legacy float adapter (T class)
        y = y + (x @ g("mag_B").to(D)) @ g("mag_A").to(D).T
    if g("row_scale") is not None:
        y = y * g("row_scale").to(D)
    if g("out_bias") is not None:
        y = y + g("out_bias").to(D)
    return y


def op_ttf_linear(ins, node, W, st, D):
    x = ins[0]
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    p, attr = node["p"], node.get("attr", {})
    if attr.get("adapter"):                                          # QTad: rD(lin(x) + up(down(x)))
        y = ttf_linear_part(x2, W, p + ".lin", attr, D)
        z = ttf_linear_part(x2, W, p + ".ad.0", attr, D)
        y = (y + ttf_linear_part(z, W, p + ".ad.1", attr, D)).to(D)
    else:
        y = ttf_linear_part(x2, W, p, attr, D)
    if len(ins) > 1 and ins[1] is not None:                         # acc
        y = (ins[1].reshape(y.shape).to(D) + y).to(D)
    return [y.float().reshape(*shp[:-1], y.shape[-1])]


def op_embed(ins, node, W, st, D):
    a, p = node.get("attr", {}), node["p"]
    idx = ins[0].long()
    if a.get("kind", "float") == "float":
        return [W[p + ".weight"][idx].float()]
    g = a["g"]                                                       # int-k: D(lo + s q), fp32 mul then add
    q = W[p + ".codes_u8"][idx].float()
    q = q.reshape(*idx.shape, -1, g)
    w = W[p + ".lo"].float()[idx][..., None] + W[p + ".s"].float()[idx][..., None] * q
    return [w.reshape(*idx.shape, -1).to(D).float()]


def _intk_w(W, p, g, D):
    q = W[p + ".codes_u8"].float()
    w = W[p + ".lo"].float()[..., None] + W[p + ".s"].float()[..., None] * q.reshape(q.shape[0], -1, g)
    return w.reshape(q.shape[0], -1).to(D)


def op_head(ins, node, W, st, D):
    a = node.get("attr", {})
    w = _intk_w(W, node["p"], a["g"], D) if a.get("kind") == "intk" else W[node["p"] + ".weight"].to(D)
    return [(ins[0].to(D) @ w.t()).float()]


def op_matmul(ins, node, W, st, D):
    """x [..., K] @ <p>.weight [N, K]^T (+ <p>.bias), the weight in its stored dtype; computed in f32"""
    p = node["p"]
    y = ins[0].float() @ W[p + ".weight"].float().t()
    if node.get("attr", {}).get("bias"):
        y = y + W[p + ".bias"].float()
    return [y]


def op_rms_norm(ins, node, W, st, D):
    x = ins[0].float()
    a = node["attr"]
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + a["eps"])
    w = W[node["p"] + ".weight"].float()
    return [y * (1 + w if a.get("offset") else w)]


def _rope_angles(a, pos, hd, dev):
    rd = a.get("dims", hd)
    inv = 1.0 / (a["theta"] ** (torch.arange(0, rd, 2, device=dev)[: rd // 2].float() / rd))
    sc = a.get("scaling")
    t = pos.float().to(dev)
    if sc and sc["type"] == "longrope":                             # angle = (pos / factor_j) * inv_j, x mscale
        f = torch.tensor(sc["factors"], dtype=torch.float32, device=dev)
        ang = torch.outer(t, 1.0 / f) * inv
        return ang, sc.get("mscale", 1.0)
    return torch.outer(t, inv), 1.0


def op_rope(ins, node, W, st, D):
    """x [..., S, H, hd], pos [S]"""
    x, pos = ins[0].float(), ins[1]
    a = node["attr"]
    hd = x.shape[-1]
    ang, ms = _rope_angles(a, pos, hd, x.device)
    c, s = (ang.cos() * ms)[:, None, :], (ang.sin() * ms)[:, None, :]
    rd = a.get("dims", hd)
    xr, xp = x[..., :rd], x[..., rd:]
    if a.get("style", "interleaved") == "interleaved":
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        y = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)
    else:
        h = rd // 2
        x1, x2 = xr[..., :h], xr[..., h:]
        y = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    return [torch.cat([y, xp], -1) if rd < hd else y]


def op_attn(ins, node, W, st, D):
    """causal: q [T, H, hd], k / v [T, Hkv, hd], pos [T], state kv (written at pos).
    bidirectional: q [..., S, H, hd], k / v [..., S, Hkv, hd] (leading batch dims), no state"""
    q, k, v = (t.float() for t in ins[:3])
    a = node.get("attr", {})
    mask = None
    if a.get("causal", True):
        pos = ins[3]
        kv = st.setdefault(a["state"], {"k": {}, "v": {}})
        for i, ps in enumerate(pos.tolist()):
            kv["k"][ps] = k[i]
            kv["v"][ps] = v[i]
        P = sorted(kv["k"])
        k = torch.stack([kv["k"][j] for j in P])
        v = torch.stack([kv["v"][j] for j in P])
        mask = torch.tensor(P, device=q.device)[None, :] <= pos.to(q.device)[:, None]     # [T, S]
    H, Hkv = q.shape[-2], k.shape[-2]
    if H != Hkv:
        k = k.repeat_interleave(H // Hkv, dim=-2)
        v = v.repeat_interleave(H // Hkv, dim=-2)
    scale = a.get("scale", 1.0 / math.sqrt(q.shape[-1]))
    sc = torch.einsum("...thd,...shd->...hts", q, k) * scale
    if mask is not None:
        sc = sc.masked_fill(~mask, float("-inf"))
    return [torch.einsum("...hts,...shd->...thd", sc.softmax(-1), v)]


def op_glu(ins, node, W, st, D):
    g, u = ins[0], ins[1]
    act = node.get("attr", {}).get("act", "silu")
    return [(F.silu(g) if act == "silu" else F.gelu(g)) * u]


def _shape(spec, x):
    """IR shape: ints, -1, or names; a name ("T", "B", ...) takes x's own dim at that position"""
    return [x.shape[i] if isinstance(d, str) else d for i, d in enumerate(spec)]


def op_reshape(ins, node, W, st, D):
    return [ins[0].reshape(_shape(node["attr"]["shape"], ins[0]))]


def op_permute(ins, node, W, st, D):
    return [ins[0].permute(node["attr"]["dims"])]


def _operand2(ins, node, W):
    """the second operand: a value, a named weight (attr w) broadcast over trailing dims, or a scalar (attr c)"""
    a = node.get("attr", {})
    if "w" in a:
        return W[a["w"]].float()
    return a["c"] if "c" in a else ins[1]


def op_add(ins, node, W, st, D):
    return [ins[0] + _operand2(ins, node, W)]


def op_sub(ins, node, W, st, D):
    return [ins[0] - _operand2(ins, node, W)]


def op_mul(ins, node, W, st, D):
    return [ins[0] * _operand2(ins, node, W)]


def op_div(ins, node, W, st, D):
    return [ins[0] / _operand2(ins, node, W)]


def op_scale(ins, node, W, st, D):
    return [ins[0] * node["attr"]["s"]]


def _unary(f):
    return lambda ins, node, W, st, D: [f(ins[0])]


def op_concat(ins, node, W, st, D):
    return [torch.cat(ins, node["attr"]["dim"])]


def op_slice(ins, node, W, st, D):
    """attr dim, start, len (None: to the end)"""
    a = node["attr"]
    x = ins[0]
    d = a["dim"] % x.ndim
    n = x.shape[d] - a["start"] if a.get("len") is None else a["len"]
    return [x.narrow(d, a["start"], n)]


def op_repeat(ins, node, W, st, D):
    """ins[0] repeated (broadcast) to the shape of ins[1] (ggml_repeat)"""
    return [ins[0].expand_as(ins[1]).clone()]


def op_const(ins, node, W, st, D):
    """the named tensor (attr w) as a value, f32, reshaped to attr shape when given"""
    a = node["attr"]
    t = W[a["w"]].float()
    return [t.reshape(a["shape"]) if "shape" in a else t]


def op_arange(ins, node, W, st, D):
    a = node["attr"]
    return [torch.arange(a.get("start", 0), a["n"], dtype=torch.int32)]


def op_timestep_embed(ins, node, W, st, D):
    """sinusoidal: t [B] -> [B, dim] = cat(sin(scale t f), cos(scale t f)), f_j = exp(-j ln(max_period) / (half - 1))"""
    a = node["attr"]
    t = ins[0].float()
    half = a["dim"] // 2
    f = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -(math.log(a.get("max_period", 10000)) / (half - 1)))
    e = a.get("scale", 1.0) * t[:, None] * f[None, :]
    return [torch.cat([e.sin(), e.cos()], -1)]


def op_snake(ins, node, W, st, D):
    """x [C, T]: x + 1 / (alpha + 1e-9) sin^2(alpha x), alpha <p>.alpha [C]"""
    al = W[node["p"] + ".alpha"].float().reshape(-1, 1)
    x = ins[0]
    return [x + (al + 1e-9).reciprocal() * torch.sin(al * x).pow(2)]


def _tail(st, a, x, n):
    """the conv state: the last n input frames of the stream so far (zeros at start); returns [C, n + T]"""
    sid = a.get("state")
    prev = st.get(sid) if sid else None
    if prev is None:
        prev = torch.zeros(x.shape[0], n, dtype=x.dtype, device=x.device)
    full = torch.cat([prev, x], -1)
    if sid:
        st[sid] = full[:, full.shape[-1] - n:]
    return full


def op_conv1d_causal(ins, node, W, st, D):
    """x [C, T] -> [Cout, T']: conv1d over (left context `ctx` frames, x); the context is the state (streaming) or
    zeros. <p>.weight [Cout, C / groups, k], <p>.bias when attr bias; act: none / silu"""
    a, p = node["attr"], node["p"]
    x = ins[0].float()
    full = _tail(st, a, x, a["ctx"])
    b = W[p + ".bias"].float() if a.get("bias") else None
    y = F.conv1d(full[None], W[p + ".weight"].float(), b, stride=a.get("stride", 1),
                 dilation=a.get("dilation", 1), groups=a.get("groups", 1))[0]
    return [F.silu(y) if a.get("act") == "silu" else y]


def op_conv_transpose1d_causal(ins, node, W, st, D):
    """x [C, T] -> [Cout, T s]: conv_transpose1d over (ctx past frames, x), output [ctx s : len - trim]"""
    a, p = node["attr"], node["p"]
    x = ins[0].float()
    c, s = a["ctx"], a["stride"]
    full = _tail(st, a, x, c)
    b = W[p + ".bias"].float() if a.get("bias") else None
    y = F.conv_transpose1d(full[None], W[p + ".weight"].float(), b, stride=s, groups=a.get("groups", 1))[0]
    return [y[:, c * s: y.shape[-1] - a["trim"]]]


def op_delta_rule(ins, node, W, st, D):
    """gated delta rule, recurrent form in fp32 (the semantic definition; a chunked prefill is a T-class variant):
    q / k [T, Hk, dk], v [T, Hv, dv], g [T, Hv] (log decay), b [T, Hv] (beta); v head h reads q / k head h // (Hv / Hk).
    qk_l2norm: q, k <- x / sqrt(|x|^2 + 1e-6); q *= scale. Per token: S <- S exp(g); S <- S + k (b (v - S^T k))^T;
    o = S^T q. State rec [Hv, dk, dv] (zeros at start)."""
    q, k, v, g, b = (t.float() for t in ins[:5])
    a = node["attr"]
    r = v.shape[1] // q.shape[1]
    if r > 1:
        q, k = q.repeat_interleave(r, 1), k.repeat_interleave(r, 1)
    if a.get("qk_l2norm", True):
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * a.get("scale", q.shape[-1] ** -0.5)
    S = st.get(a["state"])
    if S is None:
        S = torch.zeros(v.shape[1], q.shape[-1], v.shape[-1], dtype=torch.float32, device=v.device)
    out = torch.empty_like(v)
    for t in range(v.shape[0]):
        S = S * g[t].exp()[:, None, None]
        kv = (S * k[t][:, :, None]).sum(-2)
        d = (v[t] - kv) * b[t][:, None]
        S = S + k[t][:, :, None] * d[:, None, :]
        out[t] = (S * q[t][:, :, None]).sum(-2)
    st[a["state"]] = S
    return [out]


OPS = {"ttf.linear": op_ttf_linear, "ttf.embed": op_embed, "ttf.head": op_head, "matmul": op_matmul,
       "rms_norm": op_rms_norm, "rope": op_rope, "attn": op_attn, "glu": op_glu, "reshape": op_reshape,
       "permute": op_permute, "add": op_add, "sub": op_sub, "mul": op_mul, "div": op_div, "scale": op_scale,
       "silu": _unary(F.silu), "tanh": _unary(torch.tanh), "sigmoid": _unary(torch.sigmoid),
       "round": _unary(torch.round), "softplus": _unary(F.softplus), "delta_rule": op_delta_rule, "exp": _unary(torch.exp), "neg": _unary(torch.neg),
       "concat": op_concat, "slice": op_slice, "repeat": op_repeat, "const": op_const, "arange": op_arange,
       "timestep_embed": op_timestep_embed, "snake": op_snake, "conv1d_causal": op_conv1d_causal,
       "conv_transpose1d_causal": op_conv_transpose1d_causal}


class Runner:
    """runs the IR's graphs on named tensors W (torch) with state kept across calls; device: where it computes"""

    def __init__(self, ir, W, D, device="cpu"):
        self.ir, self.D, self.dev = ir, D, device
        self.W = {k: v.to(device) for k, v in W.items()}
        self.state = {}
        ks, kn = {}, {}
        for g in ir["graphs"].values():
            for n in g["nodes"]:
                if n["op"] == "ttf.linear":
                    a = n["attr"]
                    for sub, K in (a.get("Ks") or {"": a["K"]}).items():
                        ks[n["p"] + sub] = K
                elif n["op"] in ("ttf.embed", "ttf.head") and n.get("attr", {}).get("kind") == "intk":
                    kn[n["p"]] = (n["attr"]["K"], n["attr"]["bits"])
        for p, K in ks.items():
            self.W[p + ".trits_i8"] = unpack_rows5(self.W[p + ".wpacked"], K)
        for p, (K, bits) in kn.items():
            c = self.W[p + ".codes"]
            self.W[p + ".codes_u8"] = unpack_nib2(c, K) if bits <= 4 else c.to(torch.uint8)

    def reset(self):
        self.state = {}

    @torch.no_grad()
    def __call__(self, graph, **inputs):
        g = self.ir["graphs"][graph]
        vals = {k: v.to(self.dev) for k, v in inputs.items()}
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False       # the exact integer product needs true fp32
        try:
            for n in g["nodes"]:
                ins = [vals[i] if i is not None else None for i in n["in"]]
                outs = OPS[n["op"]](ins, n, self.W, self.state, self.D)
                for name, v in zip(n["out"], outs):
                    vals[name] = v
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return {o["name"]: vals[o["of"]] for o in g["outputs"]}
