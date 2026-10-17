"""Does the trained ternary Qwen lose quality to its 8-bit activation quantisation of a constant-offset input?

Every PackedTLinear quantises its input per token to int8 by the token's absmax. If the inputs carry a large constant
offset (the fp Qwen3.8-27B's residual stream does: one dimension up to ~765x the median rms, 97-99% constant), the
other dimensions get almost no resolution. Here each layer's input mean mu (measured on --calib windows of the model's
own inputs) is taken out before the quantisation and added back exactly through the layer's float map:
    y = layer(x - mu) + b,   b = row_scale * (beta T (c * mu) + A B^T (c * mu))       (identical without quantisation)
Reports the mean input stats and the loss on the first --n windows, plain vs centred (no retraining).
  python scripts/analysis/qwen_center_test.py HF_DIR PACK.pt --data data/wikiq8_val.bin --n 8
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "infer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
torch.set_num_threads(4)
import ttf_qwen                                                                     # noqa: E402
from ttf.kernel import unpack_rows                                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("hf_dir"); ap.add_argument("weights"); ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=8); ap.add_argument("--calib", type=int, default=4)
ap.add_argument("--calib_data", default="", help="default: --data windows after the scored ones")
a = ap.parse_args()
dev = "cuda"
hf, _ = ttf_qwen.build(a.hf_dir, a.weights, dev, True)
data = np.memmap(a.data, dtype=np.uint32, mode="r")
cal = np.memmap(a.calib_data, dtype=np.uint32, mode="r") if a.calib_data else data
off = 0 if a.calib_data else a.n
lins = [(n, m) for n, m in hf.named_modules() if isinstance(m, ttf_qwen.PackedTLinear)]
print(f"{len(lins)} ternary linears", flush=True)


@torch.no_grad()
def win_loss(i):
    x = torch.tensor(data[i * 2048:(i + 1) * 2048 + 1].astype(np.int64), device=dev)[None]
    lg = hf(input_ids=x[:, :-1]).logits[0]
    tot = sum(float(torch.nn.functional.cross_entropy(lg[j:j + 256].float(), x[0, 1 + j:257 + j], reduction="sum"))
              for j in range(0, lg.shape[0], 256))
    del lg
    return tot / (x.shape[1] - 1)


S, cnt, hooks = {}, {}, []
for n, m in lins:
    def hk(mod, inp, n=n):
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
        if n not in S:
            S[n] = [torch.zeros(x.shape[1], device=dev, dtype=torch.float64), 0.0, 0.0]
        S[n][0] += x.sum(0).double(); S[n][1] += float(x.pow(2).sum()); S[n][2] += x.shape[0]
        rms = x.pow(2).mean(0).sqrt()
        cnt[n] = max(cnt.get(n, 0.0), float(rms.max() / rms.median().clamp_min(1e-8)))
    hooks.append(m.register_forward_pre_hook(hk))
with torch.no_grad():
    for i in range(a.calib):
        x = torch.tensor(cal[(off + i) * 2048:(off + i + 1) * 2048].astype(np.int64), device=dev)[None]
        hf(input_ids=x)
for h in hooks:
    h.remove()
share = [float(S[n][0].pow(2).sum() / S[n][2] ** 2 / (S[n][1] / S[n][2])) for n, _ in lins]
print(f"input mean's share of E|x|^2: mean {np.mean(share):.3f}, max {np.max(share):.3f}; input outlier ratio (max dim rms "
      f"/ median): mean {np.mean(list(cnt.values())):.1f}, max {np.max(list(cnt.values())):.1f}", flush=True)
plain = [win_loss(i) for i in range(a.n)]
print(f"plain:   mean loss {np.mean(plain):.4f}: " + " ".join(f"{v:.4f}" for v in plain), flush=True)
for n, m in lins:
    mu = (S[n][0] / S[n][2]).float()
    with torch.no_grad():
        T = unpack_rows(m.wpacked, m.K).float() if m.wpacked.numel() else None
        if T is None:                        # repacked to the 2-bit layout at load
            sh = torch.tensor([0, 2, 4, 6], device=dev, dtype=torch.uint8)
            T = ((m.w2[:, :, None] >> sh) & 3).reshape(m.N, -1)[:, :m.K].float() - 1
        cm = m.col_scale.float() * mu
        beta = float(m.beta_t) if getattr(m, "beta_t", None) is not None else float(m.beta)
        b = m.row_scale.float() * (beta * (T @ cm) + m.mag_A.float() @ (m.mag_B.float().T @ cm))
        del T
    m._mu, m._b = mu.to(torch.bfloat16), b.to(torch.bfloat16)
    m.register_forward_pre_hook(lambda mod, inp: (inp[0] - mod._mu.to(inp[0].dtype),))
    m.register_forward_hook(lambda mod, inp, out: out + mod._b.to(out.dtype))
cent = [win_loss(i) for i in range(a.n)]
print(f"centred: mean loss {np.mean(cent):.4f}: " + " ".join(f"{v:.4f}" for v in cent), flush=True)
print(f"difference {np.mean(cent) - np.mean(plain):+.4f}")
