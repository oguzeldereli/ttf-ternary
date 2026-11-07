"""ttf.q38.model on a tiny random config: forward / backward run, PLE hashing equals a direct per-position recompute,
MoE equals a per-token dense loop, attention at T <= budget is causal (a prefix's outputs do not depend on later
tokens), on GPU when there is one."""
import copy, json, os, sys
import torch
from ttf.q38 import model as M

dev = "cuda" if torch.cuda.is_available() else "cpu"
real = json.load(open(sys.argv[1]))["text_config"] if len(sys.argv) > 1 else {}
t = dict(real)
t.update(hidden_size=64, hc_count=4, hc_lowrank=8, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
         head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
         num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
         vocab_size=1000, ngram_vocab_size_base=101, heads_per_ngram=2, ple_embed_dim=64, eos_token_id=999,
         bos_token_id=999, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=64,
         layer_types=["linear_attention"] * 3 + ["full_attention"], ple_layer_ids=[2], max_position_embeddings=4096)
for k, v in dict(ngram_size=3, ple_conv_kernel_size=3, hidden_act="silu", rms_norm_eps=1e-6).items():
    t.setdefault(k, v)          # (the real config.json, when given, keeps its own values)
cfg = M.text_config({"text_config": t})
torch.manual_seed(0)
m = M.build(cfg, device="cpu").to(dev)
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith("A_log"):
            p.copy_(torch.randn_like(p) * 0.1)
        elif p.dim() >= 2:
            p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
        else:
            p.copy_(torch.randn_like(p) * 0.1)
    for L in m.layers:
        if L.ple is not None:
            ng = L.ple.ple_embedding
            ng.table = torch.randn(ng.rows, ng.dim)               # CPU table, as in training
m = m.to(torch.bfloat16 if dev == "cuda" else torch.float32)
ids = torch.randint(0, 999, (2, 48), device=dev)
ids[0, 10] = 999; ids[1, 30] = 999
y = m(ids)
assert y.shape == (2, 48, 1000) and torch.isfinite(y.float()).all(), "forward"
y.float().square().mean().backward()
g = [p.grad for n, p in m.named_parameters() if p.requires_grad and p.grad is not None]
assert g and all(torch.isfinite(x.float()).all() for x in g), "backward"
print("forward/backward ok; params with grad:", len(g))

# causality: outputs at the first 20 positions do not depend on tokens 20..47
ids2 = ids.clone(); ids2[:, 20:] = torch.randint(0, 999, (2, 28), device=dev)
with torch.no_grad():
    d = (m(ids)[:, :20].float() - m(ids2)[:, :20].float()).abs().max().item()
print("causal max diff", d); assert d < 1e-2, "causality"

# PLE indices: direct recompute per position
ng = [L for L in m.layers if L.ple is not None][0].ple.ple_embedding
idx = ng.indices(ids).cpu()
eos = 999
mult = ng.mult.tolist(); sizes = ng.sizes.tolist(); offs = ng.offs.tolist()
M64 = (1 << 64) - 1
def s64(v):          # torch long arithmetic wraps mod 2^64 (two's complement); emulate
    v &= M64
    return v - (1 << 64) if v >> 63 else v
for b in range(2):
    seq = ids[b].tolist()
    for tpos in range(48):
        seg = 0
        for q in range(tpos):
            if seq[q] == eos:
                seg = q + 1
        ctx = []
        for sft in range(3):
            src = tpos - sft
            ctx.append(seq[src] if src >= seg and src >= 0 else eos)
        hcol = 0
        for gram in (2, 3):
            mix = s64(ctx[0] * mult[0])
            for p_ in range(1, gram):
                mix = s64(mix ^ s64(ctx[p_] * mult[p_]))
            for h in range(ng.hpn):
                hh = (gram - 2) * ng.hpn + h
                want = mix % sizes[hh] + offs[hh]            # python % is floor-mod, as torch.remainder
                assert idx[b, tpos, hcol] == want, (b, tpos, hcol, int(idx[b, tpos, hcol]), want)
                hcol += 1
print("PLE indices ok")

# MoE vs a per-token loop
mo = m.layers[0].mlp
x = torch.randn(1, 12, 64, device=dev, dtype=next(m.parameters()).dtype)
with torch.no_grad():
    yb = mo(x)[0].float()
    w, e = mo.route(x[0])
    ref = []
    for i in range(12):
        acc = torch.zeros(64, device=dev)
        for j in range(mo.k):
            W1 = mo.gu.weight[e[i, j]].float(); W2 = mo.dn.weight[e[i, j]].float()
            gu = W1 @ x[0, i].float()
            acc += w[i, j].float() * (W2 @ (torch.nn.functional.silu(gu[:mo.I]) * gu[mo.I:]))
        acc += torch.sigmoid(mo.shared_expert_gate(x[0, i:i+1]).float())[0] * mo.shared_expert(x[0, i:i+1]).float()[0]
        ref.append(acc)
    ref = torch.stack(ref)
print("moe rel err", ((yb - ref).norm() / ref.norm()).item())
assert ((yb - ref).norm() / ref.norm()).item() < 3e-2
print("activation of DeltaNet norm:", m.layers[0].linear_attn.norm.activation)
print("ALL OK")
