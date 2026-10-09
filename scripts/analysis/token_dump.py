"""Per-token losses of a checkpoint on the fixed evaluation windows (the same windows as scripts/eval/final_eval.py), for
paired per-token comparisons of two runs (scripts/analysis/token_gap.py).
    python scripts/analysis/token_dump.py CKPT OUT.npz [--data fw|wiki] [--n 400]
OUT.npz: tok (n, 2049) uint16 (inputs + last target), loss (n, 2048) float16 (loss of predicting tok[:, 1:]).
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scripts.eval.final_eval import load_model                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--data", default="fw", choices=["fw", "wiki"]); ap.add_argument("--n", type=int, default=400)
ap.add_argument("--bs", type=int, default=4)
a = ap.parse_args()
dev = "cuda"
path = {"fw": "data/fwedu32k_val.bin", "wiki": "data/wiki32k_val.bin"}[a.data]
data = np.memmap(path, dtype=np.uint16, mode="r")
L = 2048
n = min(a.n, (len(data) - 1) // L)
t0 = time.time()
model, ck = load_model(a.ckpt, dev)
print(f"loaded {a.ckpt} (step {ck.get('step')}, {ck['mode']}) in {time.time() - t0:.0f}s", flush=True)
tok = np.stack([np.asarray(data[i * L: i * L + L + 1]) for i in range(n)]).astype(np.uint16)
out = np.zeros((n, L), dtype=np.float16)
with torch.no_grad():
    for i in range(0, n, a.bs):
        x = torch.tensor(tok[i:i + a.bs].astype(np.int64), device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, :-1])
        l = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
        out[i:i + a.bs] = l.view(x.shape[0], L).cpu().numpy().astype(np.float16)
np.savez_compressed(a.out, tok=tok, loss=out, step=ck.get("step", -1))
print(f"wrote {a.out}: mean loss {out.astype(np.float32).mean():.4f} over {n} windows, {time.time() - t0:.0f}s", flush=True)
