"""Data parallel with the factored TTF update (--dw factored): every rank gets its own gradient factors (gy_r, x_r), the
update averages the thin products it reads from g across the ranks (ttf.update.DP_REDUCE), and every rank must then hold
exactly what one process holds after the same update from the mean gradient, g = (1 / R) sum_r gy_r^T x_r, given to it as
the concatenated factors (cat_r gy_r / R, cat_r x_r). Checked for --ttf-accum micro (M micro-batches per step) and one
micro-batch per step, several steps with firing: the accumulator U V^T and the trits. With fp32 factors (FP32=1) they
agree to fp32 summation order; with bf16 factors (the training path) the update's bf16-rounded intermediates (x (b V)
on the tensor cores) turn the ~1e-7 differences into ~1e-4, and near-threshold trits can then go the other way: the
ranks stay identical to each other in both cases (that is what data parallel needs).
  torchrun --nproc_per_node 2 scripts/tests/dp_factored_check.py      (2 ranks on one GPU share it through gloo)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import torch.distributed as dist

dist.init_process_group("gloo" if torch.cuda.device_count() < int(os.environ["WORLD_SIZE"]) else "nccl")
R, rk = dist.get_world_size(), dist.get_rank()
dev = f"cuda:{int(os.environ.get('LOCAL_RANK', 0)) % torch.cuda.device_count()}"
torch.cuda.set_device(dev)
torch.backends.cuda.matmul.allow_tf32 = False
from ttf import update as U
from ttf.layers import KernelTernaryLinear
from ttf.kernel import unpack_rows

N, K, T = 256, 320, 512
STEPS, THETA = int(os.environ.get("STEPS", "4")), float(os.environ.get("THETA", "2"))
DT = torch.float32 if os.environ.get("FP32") == "1" else torch.bfloat16


def cfg(M, s):
    return dict(r=16, lr_ratio=1.0, theta=THETA, tau=1000.0, b2=0.95, spend=1.0, store_a="",
                offload=False, theta_sched=None, progress=0.0, factored=64, start=0, step=10 + s, micro=M)


def factors(s, m, r_):
    g = torch.Generator(device=dev).manual_seed(1000 * s + 10 * m + r_)
    gy = (torch.randn(T, N, device=dev, generator=g) * 1e-2).to(DT)
    gy[:, :8] *= 10
    x = (torch.randn(T, K, device=dev, generator=g) + 0.3).to(DT)
    return gy, x


def layer():
    torch.manual_seed(7)
    l = KernelTernaryLinear(K, N, 8).to(dev); l._ts_host = False; l._ts_idx = 5
    return l


def run(M, dp):
    U.FIRE_FUSED[0] = os.environ.get("FUSED") == "1"      # --fire fused (the recipes' default)
    l, state = layer(), {}
    U.DP_REDUCE[0] = (lambda t: (dist.all_reduce(t), t.div_(R))) if dp else None
    for s in range(STEPS):
        for m in range(M):
            l._ts_final = m == M - 1
            if dp:
                fg = factors(s, m, rk)
            else:                                  # one process, the mean gradient as concatenated factors
                fs = [factors(s, m, r_) for r_ in range(R)]
                fg = (torch.cat([f[0] for f in fs]) / R, torch.cat([f[1] for f in fs]))
            U.ts_layer(l, fg, state, cfg(M, s))
    U.DP_REDUCE[0] = None
    Uu, V = U._load_pair(state[id(l)])
    return Uu @ V.T, unpack_rows(l.wpacked, K).clone()


ok = True
for M in (1, 2):
    A_dp, W_dp = run(M, True)
    A_1, W_1 = run(M, False)
    W0 = unpack_rows(layer().wpacked, K)
    moved = int((W_1 != W0).sum())
    rel = float((A_dp - A_1).norm() / A_1.norm())
    diff = int((W_dp != W_1).sum())
    full = torch.cat([W_dp.flatten().float(), A_dp.flatten()]).cpu()      # bit-identical trits and accumulator
    allc = [torch.zeros_like(full) for _ in range(R)]; dist.all_gather(allc, full)
    same = all(torch.equal(allc[0], a) for a in allc)
    tol, tolw = (5e-5, 0) if DT == torch.float32 and not U.FIRE_FUSED[0] else (3e-2, max(1, moved // 50))
    if M > 1 and tol < 1e-2:   # the turn's mass is read from the running sketch, kept in bf16 between micro-batches:
        tol, tolw = 1e-2, max(1, moved // 500)   # its roundings fall elsewhere than one process's (2.7e-3 of A here)
    good = rel < tol and diff <= tolw and moved > 0 and same
    ok &= good
    print(f"rank {rk} micro {M}: trits moved {moved}, differing from one process {diff}, accumulator rel. diff {rel:.2e}, "
          f"ranks identical {same}: {'ok' if good else 'FAIL'}", flush=True)
dist.destroy_process_group()
sys.exit(0 if ok else 1)
