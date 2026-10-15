"""Shrink and perturb (Ash & Adams 2020) a Hugging Face Llama directory before conversion: every linear weight of the
decoder layers becomes S W + N std(W) Z (Z standard normal, seeded); embedding, norms and head unchanged.
  python scripts/analysis/perturb_hf.py models/fp32_110m_hf models/fp32_110m_sp --shrink 0.8 --noise 0.3
"""
import argparse, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--shrink", type=float, default=0.8)
ap.add_argument("--noise", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
shutil.copy(os.path.join(a.src, "config.json"), a.dst)
sd = load_file(os.path.join(a.src, "model.safetensors"))
g = torch.Generator().manual_seed(a.seed)
n = 0
for k, v in sd.items():
    if k.startswith("model.layers.") and k.endswith("_proj.weight"):
        sd[k] = a.shrink * v + a.noise * v.std() * torch.randn(v.shape, generator=g); n += 1
save_file(sd, os.path.join(a.dst, "model.safetensors"))
print(f"{n} linears: W -> {a.shrink} W + {a.noise} std(W) Z, written to {a.dst}")
