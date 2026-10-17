"""Convert a master-weights checkpoint into a ternary kernel checkpoint the trainer can --resume, to branch master onto
our flip rule. Each latent W becomes T = clip(round(W / gamma), -1, 1), gamma = mean|W| (what master's forward used);
the kernel's own scale beta = 1/sqrt(K rho) is corrected by the row scales (row_scale = gamma / beta, col_scale = 1) so
the converted layer computes gamma * T x exactly as master did. Optional recipe parts start neutral: additive adapter
A = 0 (`--mag add:R`), per-head temperature 0 (`--qk_temp`). The tail (embeddings, norms) is copied; its optimizer
state starts fresh. Prints master's and the converted model's held-out loss on the same batches.
  python -m scripts.analysis.master_to_kernel MASTER_CKPT OUT_DIR [--mag add:16] [--qk_temp]
"""
import os, sys, numpy as np, torch
import ttf.model as BM
from ttf.master import build_master_transformer
from ttf.flip import build_kernel_transformer, KernelTernaryLinear, enable_rc_scales
from ttf.kernel import pack_rows

src, out = sys.argv[1], sys.argv[2]
mag = sys.argv[sys.argv.index("--mag") + 1] if "--mag" in sys.argv else ""
qk = "--qk_temp" in sys.argv
b = torch.load(src, map_location="cpu", weights_only=False)
cfg = b["cfg"]
mm = build_master_transformer(cfg, grad_checkpoint=False); mm.load_state_dict(b["model"]); mm = mm.cuda().eval()
km = build_kernel_transformer(cfg, grad_checkpoint=False, beta=True, int8=True, dw_mode="dense", g_ref=3.0)
enable_rc_scales(km)
km = km.cuda()
if mag:
    kind, r = mag.split(":")
    for l in km.modules():
        if isinstance(l, KernelTernaryLinear): l.enable_lowrank_mag(kind, int(r))
if qk:
    for a in km.modules():
        if isinstance(a, BM.Attention): a.qk_logscale = torch.nn.Parameter(torch.zeros(a.n_heads, device="cuda"))
ms = dict(mm.named_modules())
nconv = 0
with torch.no_grad():
    for name, l in km.named_modules():
        if not isinstance(l, KernelTernaryLinear): continue
        W = ms[name].weight.float()
        gamma = W.abs().mean().clamp_min(1e-5)
        T = (W / gamma).round().clamp_(-1, 1).to(torch.int8)
        l.wpacked.copy_(pack_rows(T))
        rho = (T != 0).float().mean()
        beta = 1.0 / torch.sqrt(l.K * rho)
        l.row_scale.fill_(float(gamma / beta)); l.col_scale.fill_(1.0)
        if mag: l.mag_A.zero_()
        nconv += 1
    kstate = km.state_dict()
    ncopy = 0
    for k, v in b["model"].items():                 # the float tail (embeddings, norms, head); ternary layers have no
        if k in kstate and kstate[k].shape == v.shape:   # ".weight" in the kernel model, so they never match here
            kstate[k].copy_(v.to(kstate[k].dtype)); ncopy += 1
    km.load_state_dict(kstate)
for p in km.float_tail_parameters(): p.data = p.data.float()
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
g = np.random.default_rng(3); st = g.integers(0, len(val) - 2049, 8)
X = torch.tensor(np.stack([val[s:s + 2049].astype(np.int64) for s in st])).cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    lm = np.mean([float(mm(X[i:i + 1, :-1], X[i:i + 1, 1:])[1]) for i in range(8)])
    lk = np.mean([float(km(X[i:i + 1, :-1], X[i:i + 1, 1:])[1]) for i in range(8)])
print(f"converted {nconv} layers, copied {ncopy} tail tensors; held-out loss master {lm:.4f}, converted kernel {lk:.4f} (diff {lk - lm:+.4f})")
os.makedirs(out, exist_ok=True)
torch.save({"model": km.state_dict(), "cfg": cfg, "step": int(b["step"]), "beta": True, "int8": True,
            "dw_mode": "dense", "converted_from": src}, os.path.join(out, "ckpt.pt"))
print("wrote", os.path.join(out, "ckpt.pt"), "step", b["step"])
