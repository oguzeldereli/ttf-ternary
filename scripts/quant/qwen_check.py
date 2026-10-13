"""Conversion check for Qwen3.5 / 3.8 (bitnet/qwen.py), no training: mean loss on the first N windows of a uint32 token
stream for
  hf        transformers' own Qwen3_5ForCausalLM (bf16), the reference
  fp        our wrapper with every linear in float (must match hf)
  kernel:M  the trits and scales of method M (als, gptq), the run's forward (beta, row / column scales, int8 activations)
    python scripts/quant/qwen_check.py HF_DIR --data data/wikiqw_val.bin [--n 32] [--seq 2048] [--methods als,gptq]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from bitnet.qwen import build_qwen, load_qwen, calib_hessians_qwen        # noqa: E402
from bitnet.flip import KernelTernaryLinear, enable_rc_scales             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("hf_dir"); ap.add_argument("--data", required=True); ap.add_argument("--calib_data", default="")
ap.add_argument("--n", type=int, default=32); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--methods", default="als,gptq"); ap.add_argument("--calib", type=int, default=64)
ap.add_argument("--skip_hf", action="store_true")
ap.add_argument("--device", default="cuda", help="cpu: the full-precision reference of a model too big for the GPU")
ap.add_argument("--hf_only", action="store_true", help="only the reference (transformers' own model)")
a = ap.parse_args()
dev = a.device
data = np.memmap(a.data, dtype=np.uint32, mode="r")


@torch.no_grad()
def windows_loss(fwd):
    ls = []
    for i in range(a.n):
        x = torch.tensor(data[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64), device=dev)[None]
        with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16):
            ls.append(float(fwd(x[:, :-1], x[:, 1:])))
        print(f"  window {i + 1}/{a.n}: {ls[-1]:.4f}", flush=True)
    return sum(ls) / len(ls)


if not a.skip_hf:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    t = time.time()
    m = Qwen3_5ForCausalLM.from_pretrained(a.hf_dir, dtype=torch.bfloat16).to(dev).eval()
    L = windows_loss(lambda x, y: F.cross_entropy(m(input_ids=x).logits[0].float(), y[0]))
    print(f"hf          loss {L:.4f}  ({a.n} x {a.seq}, {time.time() - t:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
if a.hf_only:
    sys.exit()

m, sw = build_qwen(a.hf_dir, "fp32", grad_checkpoint=False, device=dev)
load_qwen(m, a.hf_dir, sw)
m.eval(); print(f"fp          loss {windows_loss(lambda x, y: m(x, y)[1]):.4f}", flush=True)
del m; torch.cuda.empty_cache()

hess = calib_hessians_qwen(a.hf_dir, a.calib_data or a.data.replace("_val", "_train"), n_seq=a.calib, seq=a.seq) \
    if "gptq" in a.methods.split(",") else None
for meth in [x for x in a.methods.split(",") if x]:
    mk = lambda i, o: KernelTernaryLinear(i, o, 8, rate=0.0, beta=True, int8=True, dw_mode="dense")
    m, sw = build_qwen(a.hf_dir, "kernel", make_linear=mk, grad_checkpoint=False, device=dev)
    enable_rc_scales(m)
    t = time.time(); load_qwen(m, a.hf_dir, sw, method=meth, hess=hess)
    m.eval(); print(f"kernel:{meth:5s} loss {windows_loss(lambda x, y: m(x, y)[1]):.4f}  (converted in {time.time() - t:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
