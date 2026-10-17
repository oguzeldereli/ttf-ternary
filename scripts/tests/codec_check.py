"""kernel.pack12_scaled / unpack12_scaled against train._pack12 / _decode (the fp12s store of the TTF accumulator):
codes and decoded values must be bit-identical; then the time of both (17 Oct).
  python scripts/tests/codec_check.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import bitnet.train as T
from bitnet.kernel import pack12_scaled, unpack12_scaled

dev = "cuda"; torch.manual_seed(0)
cs = lambda X, q: X.abs().amax(0, keepdim=True).clamp_min(1e-30) / q
for N, n in ((7, 5), (300, 64), (17408, 256), (5120, 1024), (1000, 33)):
    X = torch.randn(N, n, device=dev) * torch.logspace(-8, 3, n, device=dev)[None, :]
    X[0, :3] = 0; X[1, 1] = -0.0; X[2] = 1e-38
    sc = cs(X, 32768)
    ref = T._pack12((X / sc).to(torch.float16)); new = pack12_scaled(X, sc)
    dref = (T._unpack12(ref, n)).float() * sc; dnew = unpack12_scaled(new, sc)
    same_c = torch.equal(ref, new); same_d = torch.equal(dref.view(torch.int32), dnew.view(torch.int32))
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): T._pack12((X / sc).to(torch.float16)); (T._unpack12(ref, n)).float() * sc
    torch.cuda.synchronize(); t1 = time.time()
    for _ in range(10): pack12_scaled(X, sc); unpack12_scaled(new, sc)
    torch.cuda.synchronize(); t2 = time.time()
    print(f"{N} x {n}: codes identical {same_c}, decoded identical {same_d}; encode+decode {(t1 - t0) * 100:.2f} -> {(t2 - t1) * 100:.2f} ms", flush=True)
