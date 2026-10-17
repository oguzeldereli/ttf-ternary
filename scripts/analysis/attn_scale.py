"""Can the model make attention sharp? On real validation text, per layer: rms of q and k (after RoPE), rms of
the attention logits q.k/sqrt(d), mean attention entropy, and the attention RMSNorm gain; for master and ours.

  python -m scripts.analysis.attn_scale
"""
import json, torch, numpy as np
import ttf.model as BM
from scripts.analysis.induction_heads import load
CAP = []
_orig = BM.Attention.forward


def fwd(self, x, freqs_cis):
    B, T, _ = x.shape
    q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
    k = self.wk(x).view(B, T, self.n_kv, self.head_dim)
    q, k = BM.apply_rope(q, k, freqs_cis)
    q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
    s = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
    p = s.masked_fill(mask, float("-inf")).softmax(-1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
    CAP.append((q.pow(2).mean().sqrt().item(), k.pow(2).mean().sqrt().item(),
                s.masked_select(~mask).pow(2).mean().sqrt().item(), ent))
    return _orig(self, x, freqs_cis)


BM.Attention.forward = fwd
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
g = np.random.default_rng(7)
x = torch.from_numpy(np.stack([val[s:s + 256].astype(np.int64) for s in g.integers(0, len(val) - 300, 8)])).cuda()
for name, path, kind in [("master 66M", "curve_master/ckpt_2000.pt", "master"), ("master 300M", "master_tracked/ckpt.pt", "master"),
                         ("ours 131M", "r4090_replay_11M_205M/ckpt_4000.pt", "kernel"), ("ours 300M", "lm_lowrank256_xb2_100M/ckpt.pt", "kernel")]:
    s, m, V = load("checkpoints/" + path, kind)
    CAP.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    gains = [l.attn_norm.weight.float().abs().mean().item() for l in m.layers]
    print(f"\n{name}: per layer  rms(q) rms(k) rms(logit) entropy(nats, uniform over 256 = 5.5) attn-norm gain")
    for i, ((rq, rk, rl, e), gn) in enumerate(zip(CAP, gains)):
        print(f"  L{i:2d}  {rq:6.2f} {rk:6.2f} {rl:7.2f} {e:6.2f} {gn:6.3f}")
    del m; torch.cuda.empty_cache()
