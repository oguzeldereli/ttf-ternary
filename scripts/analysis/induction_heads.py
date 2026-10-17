"""Which attention heads exist? On random token sequences repeated twice (16 x [64 + 64]), per layer and head:
  prev   attention from position t to t-1                     (previous-token head)
  dup    attention from a repeated token to its first occurrence  (t -> t-64)
  ind    attention from a repeated token to the token after its first occurrence (t -> t-63): induction
Printed per layer: the best head's prev / dup / ind score (chance ~ 1/t), for each checkpoint.

  python -m scripts.analysis.induction_heads
"""
import json, torch, numpy as np, torch.nn.functional as F
import ttf.model as BM
from ttf.master import build_master_transformer
from ttf.flip import build_kernel_transformer

CAP = []
ON = [False]                      # capture attention only while set (heads_over_time turns it on around one forward)
_orig = BM.Attention.forward


def fwd(self, x, freqs_cis):
    if not ON[0]:
        return _orig(self, x, freqs_cis)
    B, T, _ = x.shape
    q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
    k = self.wk(x).view(B, T, self.n_kv, self.head_dim)
    q, k = BM.apply_rope(q, k, freqs_cis)
    q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
    s = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf"))
    CAP.append(s.softmax(-1))                                 # [B, H, T, T]
    return _orig(self, x, freqs_cis)


BM.Attention.forward = fwd


def load(path, kind):
    b = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "master": m = build_master_transformer(b["cfg"], grad_checkpoint=False)
    elif kind == "fp32":
        m = BM.BitTransformer(b["cfg"], grad_checkpoint=False,
                              make_linear=lambda i, o: torch.nn.Linear(i, o, bias=False))
    else:
        m = build_kernel_transformer(b["cfg"], grad_checkpoint=False, beta=b.get("beta", True), int8=b.get("int8", True),
                                     dw_mode=b.get("dw_mode", "dense"), g_ref=3.0)
        for p in m.float_tail_parameters(): p.data = p.data.float()
    magA = [k for k in b["model"] if k.endswith("mag_A")]            # runs with --lowrank_mag (additive assumed
    if magA:                                                          # unless MAG_KIND=mul is set)
        import os
        from ttf.flip import KernelTernaryLinear
        r = b["model"][magA[0]].shape[1]
        for l in m.modules():
            if isinstance(l, KernelTernaryLinear):
                l.enable_lowrank_mag(os.environ.get("MAG_KIND", "add"), r)
                l.mag_cap = float(os.environ.get("MAG_CAP", 0))       # runs with --mag_cap
    if any(k.endswith("qk_logscale") for k in b["model"]):       # runs with --qk_temp
        for a in m.modules():
            if isinstance(a, BM.Attention):
                a.qk_logscale = torch.nn.Parameter(torch.zeros(a.n_heads))
    if any(k.endswith("row_scale") for k in b["model"]):          # runs with --rc_scale
        from ttf.flip import enable_rc_scales
        enable_rc_scales(m)
    missing, unexpected = m.load_state_dict(b["model"], strict=False)
    assert not any(k.endswith(("qk_logscale", "mag_A", "mag_B", "row_scale", "col_scale")) for k in missing)
    assert not unexpected, f"checkpoint keys the model does not have: {unexpected[:5]}"
    return b["step"], m.to("cuda").eval(), b["cfg"].vocab_size


if __name__ == "__main__":
    ON[0] = True
    g = np.random.default_rng(4321)
    L = [("master 66M  (val 3.35)", "curve_master/ckpt_2000.pt", "master"),
         ("master 98M  (val 3.19)", "curve_master/ckpt_3000.pt", "master"),
         ("master 164M (val 2.97)", "curve_master/ckpt_5000.pt", "master"),
         ("master 300M (val 2.75)", "master_tracked/ckpt.pt", "master"),
         ("ours 131M   (val 3.35)", "r4090_replay_11M_205M/ckpt_4000.pt", "kernel"),
         ("ours 197M   (val 3.22)", "r4090_replay_11M_205M/ckpt_6000.pt", "kernel"),
         ("ours 300M   (val 3.13)", "lm_lowrank256_xb2_100M/ckpt.pt", "kernel")]
    res = {}
    for name, path, kind in L:
        s, m, V = load("checkpoints/" + path, kind)
        rnd = torch.from_numpy(g.integers(1000, V - 1000, size=(16, 64))).cuda()
        x = torch.cat([rnd, rnd], 1)
        CAP.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(x)
        t = torch.arange(1, 128, device="cuda"); t2 = torch.arange(65, 128, device="cuda")
        rows = []
        for A in CAP:                                               # layer by layer
            A = A.mean(0)                                           # [H, T, T]
            prev = A[:, t, t - 1].mean(-1); dup = A[:, t2, t2 - 64].mean(-1); ind = A[:, t2, t2 - 63].mean(-1)
            rows.append((prev.max().item(), dup.max().item(), ind.max().item(), int(ind.argmax())))
        res[name] = rows
        print(f"\n{name}   per layer: best head's  prev / dup / ind  (induction head index)")
        print("  " + "  ".join(f"L{i}:{p:.2f}/{d:.2f}/{a:.2f}" for i, (p, d, a, _) in enumerate(rows)), flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(res, open("checkpoints/induction_heads.json", "w"))
