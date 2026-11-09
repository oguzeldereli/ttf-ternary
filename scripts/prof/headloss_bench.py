"""time and peak memory of one frozen-head loss chunk (headloss.head_loss_chunk: 512 tokens x the Qwen vocabulary,
KD weight 0.75), the torch path against the Triton kernel (9 Nov 2026).
  python scripts/prof/headloss_bench.py [--d 1024] [--n 512] [--V 248320]"""
import argparse, os, sys, time
sys.path.insert(0, os.environ.get("TTF_CORE", os.getcwd()))
import torch
import torch.nn.functional as F
from ttf import headloss as HL

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=1024); p.add_argument("--n", type=int, default=512)
p.add_argument("--V", type=int, default=248320); p.add_argument("--reps", type=int, default=20)
a = p.parse_args()
g = torch.Generator(device="cuda").manual_seed(0)
h = torch.randn(a.n, a.d, device="cuda", generator=g).to(torch.bfloat16).requires_grad_()
w = (torch.randn(a.V, a.d, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
tgt = torch.randint(0, a.V, (a.n,), device="cuda", generator=g)
tlp = F.log_softmax(torch.randn(a.n, a.V, device="cuda", generator=g) * 4, -1).to(torch.bfloat16)
for fused in (False, True):
    HL.FUSED_CEKD[0] = fused
    for _ in range(3):
        HL.head_loss_chunk(h, w, tgt, tlp, 0.75).backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    t0 = time.time()
    for _ in range(a.reps):
        HL.head_loss_chunk(h, w, tgt, tlp, 0.75).backward()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / a.reps
    print(f"headloss {'triton' if fused else 'torch '} d {a.d} n {a.n} V {a.V}: {dt * 1e3:.2f} ms/chunk, "
          f"transient peak {(torch.cuda.max_memory_allocated() - m0) / 2**30:.3f} GiB", flush=True)
