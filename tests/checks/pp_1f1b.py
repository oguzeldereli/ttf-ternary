"""--pipe 1F1B schedule check with a stand-in model (one Linear + tanh per stage, an MSE head): QwenTern.pp_train's
gradients and loss against the same layers run on one process, M micro-batches. Large hidden states (H = 5120, T = 2048:
21 MB per send, past NCCL's buffer) caught the send/send deadlock of the separate isend / recv ops (25 Oct).
  torchrun --nproc_per_node 2 tests/checks/pp_1f1b.py               (NCCL, one GPU per stage)
  BACKEND=gloo torchrun --nproc_per_node 3 tests/checks/pp_1f1b.py  (CPU, any stage count)"""
import contextlib, datetime, os, types
import torch, torch.distributed as dist
from ttf.qwen import QwenTern

be = os.environ.get("BACKEND", "nccl"); dist.init_process_group(be, timeout=datetime.timedelta(seconds=120))
r, n = dist.get_rank(), dist.get_world_size()
dev = torch.device("cuda", r) if be == "nccl" else torch.device("cpu")
if be == "nccl": torch.cuda.set_device(dev)
B, T, H = 1, int(os.environ.get("T", 2048)), int(os.environ.get("H", 5120)); M = int(os.environ.get("M", 5))
dt = torch.float32
g = torch.Generator().manual_seed(0)
W = [torch.randn(H, H, generator=g) / H ** 0.5 for _ in range(n)]
E = torch.randn(64, H, generator=g)
xs = [torch.randint(0, 64, (B, T), generator=g) for _ in range(M)]
ys = [torch.randn(B, T, H, generator=g) for _ in range(M)]

def stage(k, h):   # stage k: (embedding on 0) Linear + tanh
    return torch.tanh(h @ w[k].T)

# reference: all stages on this process
w = [x.clone().to(dev).requires_grad_() for x in W]
ref_loss = []
for i in range(M):
    h = E.to(dev)[xs[i].to(dev)]
    for k in range(n): h = stage(k, h)
    l = ((h - ys[i].to(dev)) ** 2).mean(); (l / M).backward(); ref_loss.append(float(l))
ref_g = w[r].grad.clone()

# pipeline: this rank holds stage r
w = [x.clone().to(dev).requires_grad_() for x in W]
fake = types.SimpleNamespace(pp={"rank": r, "n": n}, _ckpt_setup=lambda: None, kd_topk=None)
fake._stage = lambda x: stage(r, E.to(dev)[x] if r == 0 else x)
fake._h_meta = lambda x: ((x.shape[0], x.shape[1], H), dt)
fake._head_loss = lambda h, y: ((h - y) ** 2).mean()
order = []
batches = [(xs[i].to(dev), ys[i].to(dev), None) for i in range(M)]
last = QwenTern.pp_train(fake, batches, order.append, contextlib.nullcontext)
err = float((w[r].grad - ref_g).norm() / ref_g.norm())
ok = order == list(range(M)) and err < 1e-5 and abs(float(last) - ref_loss[-1]) < 1e-5 * max(1, abs(ref_loss[-1]))
print(f"rank {r}/{n} {be} M={M} H={H} T={T}: grad rel err {err:.2e}, last loss {float(last):.6f} vs {ref_loss[-1]:.6f}, "
      f"backward order {order}: {'OK' if ok else 'FAIL'}", flush=True)
dist.destroy_process_group()
raise SystemExit(0 if ok else 1)
