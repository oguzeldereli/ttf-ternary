"""Collapse a trained checkpoint to a pure-ternary deployment artifact.

Each BitLinear weight becomes: 5 ternary values {-1,0,1} packed per int8 byte
(3^5 = 243 <= 256), plus one float scale per matrix. That is ~1.6 bit/weight.
The float tail (embeddings, norms) is stored bf16.
"""
from __future__ import annotations
import argparse
import torch

from .model import BitTransformer


def pack_ternary(t: torch.Tensor) -> torch.Tensor:
    """Pack a {-1,0,1} int8 tensor into base-3, 5 trits per byte."""
    flat = (t.flatten().to(torch.int16) + 1)  # {0,1,2}
    pad = (-flat.numel()) % 5
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int16)])
    g = flat.view(-1, 5)
    w = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16)
    return (g * w).sum(1).to(torch.uint8)  # 0..242


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="deploy.pt")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = blob["cfg"]
    mode = blob.get("mode", "flip")
    if mode in ("kernel", "evidence"):
        from .flip import build_kernel_transformer
        model = build_kernel_transformer(mc, grad_checkpoint=False,
                                         beta=blob.get("beta", False))
    elif mode == "flip":
        from .flip import build_flip_transformer
        model, _ = build_flip_transformer(mc, grad_checkpoint=False)
    elif mode == "stateless":
        from .flip import build_stateless_transformer
        model = build_stateless_transformer(mc, grad_checkpoint=False)
    else:
        model = BitTransformer(mc, grad_checkpoint=False)
    model.load_state_dict(blob["model"], strict=False)

    from .bitlinear import BitLinear
    from .flip import TernaryFlipLinear, StatelessFlipLinear, KernelTernaryLinear
    packed = {}
    ternary_bytes = 0
    for name, m in model.named_modules():
        if isinstance(m, (BitLinear, TernaryFlipLinear, StatelessFlipLinear,
                          KernelTernaryLinear)):
            t, scale = m.ternary_weight()
            p = pack_ternary(t)
            packed[name] = {"packed": p, "scale": float(scale), "shape": tuple(t.shape)}
            ternary_bytes += p.numel()

    tail = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()
            if "latent" not in k and "row_scale" not in k and "momentum" not in k}
    tail_bytes = sum(v.numel() * 2 for v in tail.values())

    torch.save({"cfg": mc, "packed": packed, "tail": tail}, args.out)
    gib = 1024**3
    print(f"ternary packed : {ternary_bytes/gib:.3f} GiB")
    print(f"float tail bf16: {tail_bytes/gib:.3f} GiB")
    print(f"total deploy   : {(ternary_bytes+tail_bytes)/gib:.3f} GiB -> {args.out}")


if __name__ == "__main__":
    main()
