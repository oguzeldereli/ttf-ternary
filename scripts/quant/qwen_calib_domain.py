"""Does GPTQ's calibration data make the converted model forget other domains? (16 Oct)

Qwen3.5-0.8B-Base, streaming sequential conversion (ttf/qwen.py convert_qwen_streaming), calibrated on each
--calib set in turn (64 x 2048 tokens), then scored on every --eval set (first --n windows of each; fp first).
Sets are NAME=path to uint32 Qwen3.5 token files; a calib entry may join two files with '+' (half the windows each),
and NAME=als rounds without data (nearest-grid ALS, no GPTQ).
  python scripts/quant/qwen_calib_domain.py HF_DIR --calib wiki=data/wikiqw_train40.bin,chat=data/chatqw_train.bin,\
      mix=data/wikiqw_train40.bin+data/chatqw_train.bin --eval wiki=data/wikiqw_val.bin,chat=data/chatqw_val.bin
"""
import argparse, json, os, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ttf.qwen import build_qwen, load_qwen, convert_qwen_streaming     # noqa: E402
from ttf.flip import KernelTernaryLinear, enable_rc_scales            # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("hf_dir"); ap.add_argument("--calib", required=True)
ap.add_argument("--eval", required=True); ap.add_argument("--n", type=int, default=12); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--method", default="gptq"); ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"
sets = lambda s: [tuple(x.split("=", 1)) for x in s.split(",")]
EV = {}
for name, p in sets(a.eval):
    d = np.memmap(p, dtype=np.uint32, mode="r")
    EV[name] = [d[i * a.seq:(i + 1) * a.seq + 1] for i in range(min(a.n, (len(d) - 1) // a.seq))]


@torch.no_grad()
def score(m):
    out = {}
    for name, wins in EV.items():
        ls = []
        for w in wins:
            x = torch.tensor(w.astype(np.int64), device=dev)[None]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ls.append(float(m(x[:, :-1], x[:, 1:])[1]))
        out[name] = round(float(np.mean(ls)), 4)
    return out


R = {}
m, sw = build_qwen(a.hf_dir, "fp32", grad_checkpoint=False, device=dev); load_qwen(m, a.hf_dir, sw); m.eval()
R["fp"] = score(m); print("fp", R["fp"], flush=True)
del m; torch.cuda.empty_cache()
mk = lambda i, o: KernelTernaryLinear(i, o, 8, rate=0.0, beta=True, int8=True, dw_mode="dense")
tmp = tempfile.mkdtemp()
for name, p in sets(a.calib):
    if "+" in p:      # half the calibration windows from each file: interleave equal-length pieces into one file
        parts = [np.memmap(q, dtype=np.uint32, mode="r") for q in p.split("+")]
        L = min(len(x) for x in parts)
        p = os.path.join(tmp, name + ".bin"); np.concatenate([x[:L] for x in parts]).astype(np.uint32).tofile(p)
    m, sw = build_qwen(a.hf_dir, "kernel", make_linear=mk, grad_checkpoint=False, device=dev); enable_rc_scales(m); m.eval()
    if p == "als":
        load_qwen(m, a.hf_dir, sw, method="als")
    else:
        convert_qwen_streaming(m, a.hf_dir, sw, p, method=a.method, n_seq=64, seq=a.seq, log=lambda *_: None)
    R["calib_" + name] = score(m); print(f"{'als' if p == 'als' else a.method} calibrated on {name}:", R["calib_" + name], flush=True)
    del m; torch.cuda.empty_cache()
    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
