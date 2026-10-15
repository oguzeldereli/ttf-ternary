"""Residual-stream geometry of a Qwen3.5 / 3.8 model (bf16, streamed layer by layer through one GPU, as
qwen_zeroshot.StreamedFP): does it carry the large constant offset the 110M fp model has (scripts/analysis/geometry.py)?
Per block input on --n windows of a uint32 token stream: norm, outlier ratio (max / median per-dimension rms), excess
kurtosis across dimensions, the top-4 dimensions' share of |h|^2 and how much of that is constant over tokens
(mean^2 / mean of squares), the top dimensions; and per linear kind the input mean's share of E|x|^2 (what centred GPTQ
takes out).
  python scripts/analysis/qwen_resid_stats.py HF_DIR --data data/mix3_train.bin --n 2
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
torch.set_num_threads(4)
from qwen_zeroshot import StreamedFP                                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("hf_dir"); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=2)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda"
data = np.memmap(a.data, dtype=np.uint32, mode="r")
x = torch.tensor(np.stack([data[i * a.seq:(i + 1) * a.seq].astype(np.int64) for i in range(a.n)]))
m = StreamedFP(a.hf_dir)
mq, t = m.mq, m.t
stats, lin_mean = [], {}
with torch.no_grad():
    H = m.emb[x].to(dev)
    b, T = x.shape
    x0 = torch.zeros(b, T, t.hidden_size, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, b, -1)
    mk = {"config": t, "inputs_embeds": x0, "attention_mask": None, "past_key_values": None, "position_ids": pos[0]}
    masks = {"full_attention": mq.create_causal_mask(**mk), "linear_attention": mq.create_recurrent_attention_mask(**mk)}
    pe = m.hf.model.rotary_emb(x0, pos[1:]); del x0
    for li, layer in enumerate(m.hf.model.layers):
        h = H.reshape(-1, H.shape[-1]).float()
        rms = h.pow(2).mean(0).sqrt(); top = rms.argsort(descending=True)[:4]
        z = (h - h.mean(1, keepdim=True)) / h.std(1, keepdim=True).clamp_min(1e-8)
        sq = h[:, top].pow(2)
        stats.append({"layer": li, "norm": float(h.norm(dim=1).mean()), "outlier": float(rms.max() / rms.median()),
                      "kurtosis": float((z ** 4).mean() - 3), "top4_share": float(sq.sum(1).mean() / h.pow(2).sum(1).mean()),
                      "top4_const": float((h[:, top].mean(0).pow(2).sum()) / sq.mean(0).sum()), "top": top.tolist()})
        hooks = []
        for n_, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                def hk(mod_, inp, key=(li, n_)):
                    xx = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    lin_mean[key] = float(xx.mean(0).pow(2).sum() / xx.pow(2).sum(1).mean())
                hooks.append(mod.register_forward_pre_hook(hk))
        for n_, p in layer.named_parameters():
            p.data = m.host[li][n_].to(dev, non_blocking=True)
        o = layer(H, position_embeddings=pe, attention_mask=masks[t.layer_types[li]], position_ids=pos[0])
        H = o[0] if isinstance(o, tuple) else o
        for n_, p in layer.named_parameters():
            p.data = m.host[li][n_]
        for hh in hooks:
            hh.remove()
        s = stats[-1]
        print(f"block {li:2d} ({t.layer_types[li][:6]}): norm {s['norm']:8.1f} outlier {s['outlier']:6.1f} kurtosis "
              f"{s['kurtosis']:7.1f} top4 share {s['top4_share']:.2f} const {s['top4_const']:.2f} dims {s['top']}", flush=True)
kinds = {}
for (li, n_), v in lin_mean.items():
    kinds.setdefault(n_.split(".")[-1], []).append(v)
print("input mean's share of E|x|^2 by linear: " + ", ".join(f"{k} {np.mean(v):.3f} (max {np.max(v):.3f})" for k, v in kinds.items()))
if a.out:
    json.dump({"blocks": stats, "lin_mean_share": {f"{k[0]}.{k[1]}": v for k, v in lin_mean.items()}}, open(a.out, "w"), indent=1)
