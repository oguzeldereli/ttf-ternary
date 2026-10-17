"""How efficient is one look-ahead flip step, and how does that change over training?

For each checkpoint (same flip rate everywhere, one step, kernel path):
  pred_train   first-order loss change the kept flips intend, from the training batch's
               gradient:  sum_l <g_l, Delta_l>  (g = dL/dq, trit units)
  pred_val     the same flips scored by a held-out gradient (2 val batches): the part of
               the intended change that is real signal rather than batch noise
  realized     measured held-out loss change after applying the kept flips
  signal  = pred_val / pred_train   (selection noise: fraction of the intent that is real)
  survive = realized / pred_val     (curvature/interaction: fraction the loss delivers)
  eff     = realized / pred_train   (both)

  python efficiency_test.py ckpt1.pt ckpt2.pt ...
"""
import sys, numpy as np, torch
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf.kernel import fused_flip, unpack_rows, pack_rows, trit_beta
from ttf.train import get_batch, gpu_temp
import time


def cool(hi=82, lo=76):
    """thermal guard: wait while the GPU is hot (sustained load has crashed the machine)"""
    t = gpu_temp()
    if t is not None and t >= hi:
        while t is not None and t > lo:
            time.sleep(5); t = gpu_temp()
dev = "cuda"
RATE, G_REF, BS, SEQ, EVAL_B = 0.02, 3.0, 16, 2048, 8
# FRAC < 1 applies only a random fraction of the kept flips (linearity check: a small
# enough step must realize ~ its first-order prediction)
import os
FRAC = float(os.environ.get("FRAC", "1"))

train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
xa, ya = get_batch(train, BS, SEQ, dev, torch.Generator().manual_seed(1001))
gv = torch.Generator().manual_seed(2002)
EV = [get_batch(val, BS, SEQ, dev, gv) for _ in range(EVAL_B)]
GV = [get_batch(val, BS, SEQ, dev, torch.Generator().manual_seed(3003 + i)) for i in range(2)]


def run(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    m = build_kernel_transformer(b["cfg"], grad_checkpoint=True, beta=b.get("beta", True),
                                 int8=b.get("int8", True), dw_mode=b.get("dw_mode", "dense"),
                                 g_ref=G_REF)
    m.load_state_dict(b["model"], strict=False)
    m = m.to(dev).train()
    Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
    Q0 = [unpack_rows(l.wpacked, l.K).clone() for l in Ls]
    B0 = [trit_beta(l.wpacked, l.K).item() for l in Ls]

    def set_q(Q):
        for l, q in zip(Ls, Q): l.wpacked.copy_(pack_rows(q))

    def grad(batches):
        acc = None
        for x, y in batches:
            cool()
            for l in Ls: l.capture = True
            for p in m.parameters(): p.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, y)[1]
            loss.backward()
            gs = [l.gw.clone() for l in Ls]
            for l in Ls: l.capture = False; l.gw = None
            acc = gs if acc is None else [a + g for a, g in zip(acc, gs)]
        for p in m.parameters(): p.grad = None
        return [a / len(batches) for a in acc]

    def held():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = []
            for x, y in EV:
                cool(); out.append(m(x, y)[1].item())
            return float(np.mean(out))

    set_q(Q0); L0 = held()
    g = grad([(xa, ya)]); gval = grad(GV)
    D = []
    for i, (l, gw) in enumerate(zip(Ls, g)):
        wp = pack_rows(Q0[i])
        fused_flip(wp, gw, RATE, G_REF, 777 + i, gmean=gw.abs().mean().item())
        D.append((unpack_rows(wp, l.K) - Q0[i]).to(torch.int8))
    set_q([q + d for q, d in zip(Q0, D)])
    g1 = grad([(xa, ya)])
    K = [(d != 0) & (d.float() * (a + c) < 0) for d, a, c in zip(D, g, g1)]
    if FRAC < 1:
        gsub = torch.Generator(device=dev).manual_seed(55)
        K = [k & (torch.rand(k.shape, device=dev, generator=gsub) < FRAC) for k in K]
    Dk = [d * k.to(torch.int8) for d, k in zip(D, K)]
    set_q([q + d for q, d in zip(Q0, Dk)]); L1 = held()
    # l.gw is already dL/dq (beta is applied inside the kernel path): no extra beta
    pt = sum((gw * d.float()).sum().item() for gw, d in zip(g, Dk))
    pv = sum((gw * d.float()).sum().item() for gw, d in zip(gval, Dk))
    n = sum(int(k.sum()) for k in K)
    real = L1 - L0
    return dict(L0=L0, kept=n, pred_train=pt, pred_val=pv, realized=real,
                signal=pv / pt, survive=real / pv if pv else float("nan"), eff=real / pt,
                per_flip=real / max(n, 1))


print(f"rate {RATE}, frac {FRAC}, one look-ahead step, held-out on {EVAL_B}x{BS}x{SEQ} tokens", flush=True)
print(f"{'checkpoint':44s} {'L0':>6s} {'kept':>9s} {'pred_train':>10s} {'pred_val':>9s} "
      f"{'realized':>9s} {'signal':>7s} {'survive':>8s} {'eff':>6s} {'per flip':>9s}", flush=True)
for c in sys.argv[1:]:
    r = run(c)
    print(f"{c:44s} {r['L0']:6.3f} {r['kept']:9d} {r['pred_train']:+10.4f} {r['pred_val']:+9.4f} "
          f"{r['realized']:+9.4f} {r['signal']:7.2f} {r['survive']:8.2f} {r['eff']:6.2f} "
          f"{r['per_flip']:+9.2e}", flush=True)
    torch.cuda.empty_cache()
