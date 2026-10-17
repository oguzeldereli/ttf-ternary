"""LeanAdam.BATCHED against the per-tensor loop (17 Oct): 1-D fp16 / bf16 parameters and 2-D bf16 ones of a few shapes,
several steps; with the stochastic rounding's noise fixed (rand_like -> 0.5, so both paths draw the same numbers) the
parameters and states must agree to fp32 rounding; then the time of one step of each for a 27B-like parameter set
(400 x 3 fp16 vectors, 800 bf16 adapters of 6 shapes).
  python scripts/tests/lean_batched_check.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from ttf.opt_lean import LeanAdam

dev = "cuda"


def make(seed, shapes):
    g = torch.Generator(device=dev).manual_seed(seed)
    ps = [torch.nn.Parameter((torch.randn(*s, device=dev, generator=g) * 0.1).to(dt)) for s, dt in shapes]
    return ps


def grads(ps, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    for p in ps:
        p.grad = (torch.randn(p.shape, device=dev, generator=g) * 1e-3).to(p.dtype)


small = [((300,), torch.float16), ((300,), torch.float16), ((77,), torch.bfloat16), ((64, 16), torch.bfloat16),
         ((64, 16), torch.bfloat16), ((40, 16), torch.bfloat16)]
_rl = torch.rand_like
torch.rand_like = lambda x, *a, **k: torch.full_like(x, 0.5)
res = {}
for batched in (False, True):
    LeanAdam.BATCHED = batched
    ps = make(0, small)
    opt = LeanAdam(ps, lr=1e-2, weight_decay=0.1)
    for s in range(5):
        grads(ps, 100 + s); opt.step()
    res[batched] = ([p.detach().float().clone() for p in ps],
                    [{k: v.float().clone() for k, v in opt.state[p].items() if torch.is_tensor(v)} for p in ps])
torch.rand_like = _rl
dp = max(float((a - b).abs().max()) for a, b in zip(res[False][0], res[True][0]))
ds = max(float((res[False][1][i][k] - res[True][1][i][k]).abs().max()) for i in range(len(small)) for k in res[False][1][i])
print(f"max |param diff| {dp:.3e}, max |state diff| {ds:.3e} (5 steps, noise fixed)", flush=True)

big = [((5120,), torch.float16)] * 1200 + [((17408, 16), torch.bfloat16)] * 200 + [((5120, 16), torch.bfloat16)] * 400 \
    + [((6144, 16), torch.bfloat16)] * 200
for batched in (False, True):
    LeanAdam.BATCHED = batched
    ps = make(1, big); opt = LeanAdam(ps, lr=1e-3, weight_decay=0.1)
    grads(ps, 7); opt.step(); torch.cuda.synchronize()
    ts = []
    for s in range(3):
        grads(ps, 8 + s); torch.cuda.synchronize(); t0 = time.time(); opt.step(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    print(f"{'batched' if batched else 'per-tensor'}: {min(ts) * 1e3:.1f} ms per step ({len(big)} parameters)", flush=True)
