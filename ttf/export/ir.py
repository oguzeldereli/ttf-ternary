"""ttf.export.ir: the IR's Python reference.

Every IR op as plain torch: the semantic definition, and a reference runner (`Runner`) that executes a graph from a
.ttf file's IR + tensors with explicit state (KV cache per `kv` id) and explicit positions, prefill and decode with the
same graph. It is what the C++ runtime is tested against and what proves an exporter's IR before the runtime exists.
Activations are f32 except where the Arithmetic asks for D (TTF parts: input rounded to D at node entry).
"""
import json
import math
import torch
import torch.nn.functional as F

FLOAT_PATH = [False]
DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32, "f16": torch.float16}


def unpack_rows5(packed, K):
    """rows5 layout: uint8 [N, ceil(K/5)] base-3, digit j of byte i = trit 5 i + j (+1) -> int8 [N, K]"""
    p = packed.to(torch.int16)
    digs = []
    for _ in range(5):
        digs.append(p % 3); p = p // 3
    t = torch.stack(digs, -1).reshape(packed.shape[0], -1)[:, :K]
    return (t - 1).to(torch.int8)


def ttf_linear_part(x, W, p, attr, D):
    """Arithmetic steps 1-8 on one KernelTernaryLinear part; x f32 [T, K] -> f32 [T, N]"""
    g = lambda n: W.get(f"{p}.{n}")
    x = x.to(D)                                                      # node entry: rounded to D
    if g("in_mean") is not None:
        x = x - g("in_mean").to(D)
    if g("col_scale") is not None:
        x = x * g("col_scale").to(D)
    qp = 2 ** (attr.get("act_bits", 8) - 1) - 1
    amax = x.abs().amax(-1, keepdim=True)
    a = torch.maximum(amax, torch.tensor(1e-5, dtype=D)).float()
    s = a * torch.tensor(1.0 / qp, dtype=torch.float32)
    q = torch.round(x.float() / s).clamp_(-qp - 1, qp)
    T = W[f"{p}.trits_i8"]                                           # unpacked once at load
    if FLOAT_PATH[0]:     # debug: the TTF product in float, no quantiser (the float-path end-to-end test)
        y = (x.double() @ T.double().t() * g("beta").double()).to(D)
        return _post(y, x, g, D)
    acc = (q.double() @ T.double().t())
    t = ((acc.float() * s) * g("beta").float()).to(torch.bfloat16)
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
    x2 = x.reshape(-1, shp[-1]).float()
    p, attr = node["p"], node.get("attr", {})
    if attr.get("adapter"):
        y = ttf_linear_part(x2, W, p + ".lin", attr, D)
        z = ttf_linear_part(x2, W, p + ".ad.0", attr, D)
        y = (y + ttf_linear_part(z.float(), W, p + ".ad.1", attr, D)).to(D)
    else:
        y = ttf_linear_part(x2, W, p, attr, D)
    if len(ins) > 1 and ins[1] is not None:                         # acc
        y = (ins[1].reshape(y.shape).to(D) + y).to(D)
    return [y.float().reshape(*shp[:-1], y.shape[-1])]


def op_embed(ins, node, W, st, D):
    a = node.get("attr", {})
    if a.get("kind", "float") == "float":
        return [W[node["p"] + ".weight"][ins[0].long()].float()]
    raise NotImplementedError("int-k embed: TODO")


def op_head(ins, node, W, st, D):
    w = W[node["p"] + ".weight"].to(D)
    return [(ins[0].to(D) @ w.t()).float()]


def op_rms_norm(ins, node, W, st, D):
    x = ins[0].float()
    eps = node["attr"]["eps"]
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return [y * W[node["p"] + ".weight"].float()]


def op_rope(ins, node, W, st, D):
    x, pos = ins[0].float(), ins[1]
    a = node["attr"]
    hd = x.shape[-1]
    f = 1.0 / (a["theta"] ** (torch.arange(0, hd, 2)[: hd // 2].float() / hd))
    ang = torch.outer(pos.float(), f)                                # [T, hd/2]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    if a.get("style", "interleaved") == "interleaved":
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return [torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)]
    h = hd // 2
    x1, x2 = x[..., :h], x[..., h:]
    return [torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)]


def op_attn(ins, node, W, st, D):
    q, k, v, pos = (t.float() if t.is_floating_point() else t for t in ins[:4])
    a = node.get("attr", {})
    if a.get("causal", True):
        kv = st.setdefault(a["state"], {"k": {}, "v": {}})
        for i, ps in enumerate(pos.tolist()):
            kv["k"][ps] = k[i]; kv["v"][ps] = v[i]
        P = sorted(kv["k"])
        K_ = torch.stack([kv["k"][j] for j in P]); V_ = torch.stack([kv["v"][j] for j in P])
        kpos = torch.tensor(P)
        mask = kpos[None, :] <= pos[:, None]                         # [T, S]
    else:
        K_, V_, mask = k, v, None
    H, Hkv = q.shape[1], K_.shape[1]
    if H != Hkv:
        K_ = K_.repeat_interleave(H // Hkv, dim=1); V_ = V_.repeat_interleave(H // Hkv, dim=1)
    scale = a.get("scale", 1.0 / math.sqrt(q.shape[-1]))
    sc = torch.einsum("thd,shd->hts", q, K_) * scale
    if mask is not None:
        sc = sc.masked_fill(~mask[None], float("-inf"))
    return [torch.einsum("hts,shd->thd", sc.softmax(-1), V_)]


def op_glu(ins, node, W, st, D):
    g, u = ins[0], ins[1]
    act = node.get("attr", {}).get("act", "silu")
    return [(F.silu(g) if act == "silu" else F.gelu(g)) * u]


def op_reshape(ins, node, W, st, D):
    shp = [ins[0].shape[0] if d == "T" else d for d in node["attr"]["shape"]]
    return [ins[0].reshape(shp)]


def _operand2(ins, node, W):
    """the second operand: a value, or a named weight (attr w) broadcast over trailing dims"""
    return W[node["attr"]["w"]].float() if "w" in node.get("attr", {}) else ins[1]


def op_add(ins, node, W, st, D):
    return [ins[0] + _operand2(ins, node, W)]


def op_mul(ins, node, W, st, D):
    return [ins[0] * _operand2(ins, node, W)]


OPS = {"ttf.linear": op_ttf_linear, "ttf.embed": op_embed, "ttf.head": op_head, "rms_norm": op_rms_norm,
       "rope": op_rope, "attn": op_attn, "glu": op_glu, "reshape": op_reshape, "add": op_add, "mul": op_mul}


class Runner:
    """runs one IR graph on named tensors W (torch, CPU) with state kept across calls"""
    def __init__(self, ir, W, D):
        self.ir, self.W, self.D = ir, W, D
        self.state = {}
        for k in list(W):
            if k.endswith(".wpacked"):
                p = k[: -len(".wpacked")]
                self.W[p + ".trits_i8"] = unpack_rows5(W[k], self._K(p))

    def _K(self, p):
        for g in self.ir["graphs"].values():
            for n in g["nodes"]:
                if n["op"] == "ttf.linear":
                    for sub in ("", ".lin", ".ad.0", ".ad.1"):
                        if n["p"] + sub == p:
                            return n["attr"]["Ks"][sub] if "Ks" in n["attr"] else n["attr"]["K"]
        raise KeyError(p)

    @torch.no_grad()
    def __call__(self, graph, **inputs):
        g = self.ir["graphs"][graph]
        vals = dict(inputs)
        for n in g["nodes"]:
            ins = [vals[i] if i is not None else None for i in n["in"]]
            outs = OPS[n["op"]](ins, n, self.W, self.state, self.D)
            for name, v in zip(n["out"], outs):
                vals[name] = v
        return {o["name"]: vals[o["of"]] for o in g["outputs"]}
