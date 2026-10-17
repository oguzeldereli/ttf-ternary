"""Gradient SNR per layer: split a batch in half, compare the two weight gradients.

Sign agreement is 50% for pure noise and 100% for a noise-free gradient. For a
per-entry model g = mu + eps with independent halves, the cosine between halves
is c = |mu|^2 / (|mu|^2 + var_half), so the full-batch SNR is 2c/(1-c).

  python3 diag_snr.py --ckpt checkpoints/wiki_mf/ckpt.pt --data data/wiki_gpt2_20m_train.bin
"""
import argparse, numpy as np, torch
from ttf.layers import build_kernel_transformer, KernelTernaryLinear
from ttf.model import BitTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default="data/wiki_gpt2_20m_train.bin")
ap.add_argument("--batch", type=int, default=16, help="full batch; split into halves")
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--grad_checkpoint", type=int, default=1)
ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
dev = "cuda"

blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
mc = blob["cfg"]; mode = blob.get("mode", "kernel")
model = build_kernel_transformer(mc, grad_checkpoint=bool(args.grad_checkpoint),
                                 evidence=mode == "evidence", ev_bits=3,
                                 beta=blob.get("beta", False), int8=blob.get("int8", False),
                                 dw_mode=blob.get("dw_mode", "dense"),
                                 int8_dx=blob.get("int8_dx", False))
for p in model.float_tail_parameters():
    p.data = p.data.to(torch.bfloat16)
model.load_state_dict(blob["model"])
model = model.to(dev).train()
layers = [(n, m) for n, m in model.named_modules() if isinstance(m, KernelTernaryLinear)]
for _, m in layers:
    m.capture = True
print(f"{args.ckpt}: mode {mode} beta {blob.get('beta', False)} int8 {blob.get('int8', False)} step {blob.get('step')} "
      f"| {len(layers)} ternary layers | batch {args.batch} (halves of {args.batch//2}) seq {args.seq}")

data = np.memmap(args.data, dtype=np.uint16, mode="r")
gen = torch.Generator().manual_seed(args.seed)


def grads(bs):
    ix = torch.randint(len(data) - args.seq - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + args.seq].astype(np.int64)) for i in ix]).to(dev)
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.seq].astype(np.int64)) for i in ix]).to(dev)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    return {n: m.gw.clone() for n, m in layers}, loss.item()


acc = {n: [] for n, _ in layers}
for r in range(args.repeats):
    ga, la = grads(args.batch // 2)
    gb, lb = grads(args.batch // 2)
    for n, _ in layers:
        a, b = ga[n], gb[n]
        nz = (a != 0) & (b != 0)
        agree = ((torch.sign(a) == torch.sign(b)) & nz).sum() / nz.sum()
        cos = (a * b).sum() / (a.norm() * b.norm())
        k = max(1, int(0.01 * a.numel()))
        top = (a.abs() + b.abs()).flatten().topk(k).indices          # strongest 1%
        agree_top = (torch.sign(a.flatten()[top]) == torch.sign(b.flatten()[top])).float().mean()
        acc[n].append(torch.stack([agree, cos, agree_top]))
print(f"  (loss on these batches ~ {la:.3f})")

rows = []
for n, _ in layers:
    agree, cos, agree_top = torch.stack(acc[n]).mean(0).tolist()
    snr2 = max(0.0, 2 * cos / (1 - cos)) if cos < 1 else float("inf")
    rows.append((n, agree, agree_top, cos, snr2 ** 0.5))
print(f"\n{'layer':<28}{'sign agree':>11}{'top-1%':>9}{'cosine':>9}{'SNR(full batch)':>17}")
for n, a, at, c, s in rows:
    print(f"{n:<28}{a*100:>10.2f}%{at*100:>8.1f}%{c:>9.4f}{s:>17.3f}")
a = np.mean([r[1] for r in rows]); c = np.mean([r[3] for r in rows])
print(f"\nmean sign agreement {a*100:.2f}% (50% = pure noise) | mean cosine {c:.4f} "
      f"| implied full-batch SNR {max(0.0, 2*c/(1-c))**0.5:.3f}")
