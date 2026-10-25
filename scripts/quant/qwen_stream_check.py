"""Checks for the streaming conversion (ttf/qwen.py convert_qwen_streaming) on a small Qwen3.5:
  1. stepping the decoder layer by layer with the captured per-layer arguments gives the model's own loss (fp)
  2. the streaming, sequential GPTQ conversion's loss before training (against the one-shot GPTQ of qwen_check.py)
    python scripts/quant/qwen_stream_check.py HF_DIR --data data/wikiqw_val.bin --calib_data data/wikiqw_train.bin
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ttf.qwen import build_qwen, load_qwen, layer_inputs, convert_qwen_streaming   # noqa: E402
from ttf.layers import KernelTernaryLinear, enable_rc_scales                          # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("hf_dir"); ap.add_argument("--data", required=True)
ap.add_argument("--calib_data", required=True); ap.add_argument("--n", type=int, default=16); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--methods", default="gptq", help="comma list of streaming methods (gptq, gptqc: centred GPTQ)")
ap.add_argument("--skip_fp", action="store_true")
a = ap.parse_args(); dev = "cuda"
data = np.memmap(a.data, dtype=np.uint32, mode="r")
wins = [data[i * a.seq:(i + 1) * a.seq + 1] for i in range(a.n)]


@torch.no_grad()
def loss_model(m):
    ls = []
    for w in wins:
        x = torch.tensor(w.astype(np.int64), device=dev)[None]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ls.append(float(m(x[:, :-1], x[:, 1:])[1]))
    return sum(ls) / len(ls)


@torch.no_grad()
def loss_stepped(m):
    H, KW = layer_inputs(m, [w[:-1] for w in wins], 1, dev)
    ls = []
    for h, w in zip(H, wins):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for l, kw in zip(m.hf.model.layers, KW):
                h = l(h, **kw); h = h[0] if isinstance(h, tuple) else h
            z = F.linear(m.hf.model.norm(h), m.hf.lm_head.weight.to(h.dtype)).float()[0]
        ls.append(float(F.cross_entropy(z, torch.tensor(w[1:].astype(np.int64), device=dev))))
    return sum(ls) / len(ls)


if not a.skip_fp:
    m, sw = build_qwen(a.hf_dir, "fp32", grad_checkpoint=False, device=dev); load_qwen(m, a.hf_dir, sw); m.eval()
    print(f"fp: model forward {loss_model(m):.4f}, stepped layer by layer {loss_stepped(m):.4f}", flush=True)
    del m; torch.cuda.empty_cache()
mk = lambda i, o: KernelTernaryLinear(i, o, 8, beta=True)
for meth in a.methods.split(","):
    m, sw = build_qwen(a.hf_dir, "kernel", make_linear=mk, grad_checkpoint=False, device=dev); enable_rc_scales(m); m.eval()
    t = time.time(); convert_qwen_streaming(m, a.hf_dir, sw, a.calib_data, method=meth, n_seq=64, seq=a.seq)
    print(f"streaming sequential {meth}: loss {loss_model(m):.4f} (converted in {time.time() - t:.0f}s; one-shot gptq 4.9580 on 32 windows)", flush=True)
    del m; torch.cuda.empty_cache()
