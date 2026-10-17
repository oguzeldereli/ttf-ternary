"""Per-layer conversion error of ALS and GPTQ trits for a Hugging Face Llama: weight-space |W - a c T| / |W| and
output-space |X (W - a c T)^T|^2 / |X W^T|^2 (64 x 2048 calibration tokens), sorted by GPTQ's output error.
    python scripts/quant/gptq_layer_err.py HF_DIR [--calib 64] [--top 15]
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ttf.hf_init import calib_hessians, load_hf_tensors, ternarize, ternarize_gptq, _rel_err, _out_err, _unpermute, _LIN, hf_config

ap = argparse.ArgumentParser(); ap.add_argument("hf_dir"); ap.add_argument("--calib", type=int, default=64)
ap.add_argument("--top", type=int, default=15); ap.add_argument("--data", default="data/wiki32k_train.bin")
a = ap.parse_args()
H = calib_hessians(a.hf_dir, a.data, n_seq=a.calib)
sd = load_hf_tensors(a.hf_dir); c = hf_config(a.hf_dir)
rows = []
for i in range(c["num_hidden_layers"]):
    for hf, (sub, name) in _LIN.items():
        W = sd[f"model.layers.{i}.{hf}.weight"].cuda()
        if name == "wq": W = _unpermute(W, c["num_attention_heads"])
        if name == "wk": W = _unpermute(W, c.get("num_key_value_heads", c["num_attention_heads"]))
        Hl = H[(i, name)].cuda()
        T, al, cl = ternarize(W, "als")
        e_w0, e_o0 = _rel_err(W, T, al, cl), _out_err(W, T, al, cl, Hl)
        T2, a2, c2 = ternarize_gptq(W, Hl, al.clone(), cl.clone())
        d = torch.diagonal(Hl)
        rows.append((i, name, e_w0, e_o0, _rel_err(W, T2, a2, c2), _out_err(W, T2, a2, c2, Hl),
                     float(d.max() / d.median())))
        del W, Hl
rows.sort(key=lambda r: -r[5])
print("layer type  ALS weight / output  GPTQ weight / output  max/median input energy")
for r in rows[:a.top]:
    print(f"{r[0]:3d} {r[1]:6s}  {r[2]:.3f} / {r[3]:.4f}   {r[4]:.3f} / {r[5]:.4f}   {r[6]:.0f}")
print("GPTQ output error worse than ALS in", sum(r[5] > r[3] for r in rows), "of", len(rows), "layers")
