"""Pack a TTF-converted Qwen3.5 / 3.8 model for release: the trainer's checkpoint (or slim27.py's weights) -> one file
with the trits packed 5 per byte (as trained), the small float parts in 16 bits, and the token embedding and output
head (the two 248k x d matrices, about half the bytes in bf16) in B-bit groups of G along the hidden dimension
(asymmetric: per group a fp16 scale and minimum; B = 4 by default, 8 for the near-lossless file; 16 keeps them bf16).

  python scripts/infer/pack_ttf.py CKPT OUT.pt [--emb_bits 4] [--head_bits 4] [--group 64]

ttf_qwen.py loads either form (grouped matrices are expanded to bf16 at load: the same memory and speed as bf16, a
smaller file)."""
import argparse
import sys

import torch


def quant_groups(W, bits, group):
    """W [R, C] -> (codes uint8 [R, C * bits / 8], scale fp16 [R, C / group], min fp16 [R, C / group])"""
    R, C = W.shape
    assert C % group == 0 and bits in (4, 8)
    qmax = (1 << bits) - 1
    codes, scales, mins = [], [], []
    for r0 in range(0, R, 8192):                        # row blocks: no fp32 copy of the whole matrix
        w = W[r0:r0 + 8192].float().view(-1, C // group, group)
        lo = w.amin(-1, keepdim=True); hi = w.amax(-1, keepdim=True)
        sc = ((hi - lo) / qmax).clamp_min(1e-8)
        sc16, lo16 = sc.half(), lo.half()                # quantise against the stored (fp16) scale and minimum
        q = ((w - lo16.float()) / sc16.float()).round().clamp_(0, qmax).to(torch.uint8).view(w.shape[0], C)
        if bits == 4:
            q = q[:, 0::2] | (q[:, 1::2] << 4)
        codes.append(q); scales.append(sc16.view(w.shape[0], -1)); mins.append(lo16.view(w.shape[0], -1))
    return torch.cat(codes), torch.cat(scales), torch.cat(mins)


def dequant_groups(q, sc, mn, bits, group, dtype=torch.bfloat16):
    """inverse of quant_groups (row blocks), -> dtype [R, C]"""
    R = q.shape[0]
    C = q.shape[1] * (2 if bits == 4 else 1)
    out = torch.empty(R, C, dtype=dtype, device=q.device)
    for r0 in range(0, R, 8192):
        c = q[r0:r0 + 8192]
        if bits == 4:
            c = torch.stack([c & 15, c >> 4], -1).view(c.shape[0], C)
        w = c.float().view(c.shape[0], C // group, group) * sc[r0:r0 + 8192].float()[..., None] \
            + mn[r0:r0 + 8192].float()[..., None]
        out[r0:r0 + 8192] = w.view(c.shape[0], C).to(dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--emb_bits", type=int, default=4); ap.add_argument("--head_bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("model", ck)
    out, nbytes = {}, {"trits": 0, "grouped": 0, "float": 0}
    for k, v in sd.items():
        if k.endswith("frozen_scale") or not k.startswith("hf."):   # QwenTern also registers hf.model.layers as
            continue                                                 # 'layers.': the same tensors under a second name
        name = k[3:] if k.startswith("hf.") else k
        bits = a.emb_bits if name == "model.embed_tokens.weight" else a.head_bits if name == "lm_head.weight" else 0
        if bits and bits < 16:
            q, sc, mn = quant_groups(v, bits, a.group)
            out[k + ".q"] = q; out[k + ".qscale"] = sc; out[k + ".qmin"] = mn
            out[k + ".qmeta"] = torch.tensor([bits, a.group, v.shape[0], v.shape[1]])
            err = (dequant_groups(q[:4096], sc[:4096], mn[:4096], bits, a.group).float() - v[:4096].float()).norm() \
                / v[:4096].float().norm()
            print(f"{name}: {bits}-bit groups of {a.group}, relative error {err:.4f} (first 4096 rows)", flush=True)
            nbytes["grouped"] += q.numel() + 2 * sc.numel() * 2
        elif v.dtype == torch.uint8:
            out[k] = v.contiguous(); nbytes["trits"] += v.numel()
        elif v.is_floating_point():
            out[k] = v.to(torch.bfloat16 if v.dtype in (torch.float32, torch.bfloat16) and "scale" not in k else torch.float16
                          if "scale" in k else v.dtype).contiguous()
            nbytes["float"] += out[k].numel() * out[k].element_size()
        else:
            out[k] = v
    torch.save({"model": out, "format": "ttf-packed-1", "init_hf": ck.get("init_hf", ""), "step": ck.get("step")}, a.out)
    tot = sum(nbytes.values())
    print(f"{a.out}: trits {nbytes['trits'] / 1e9:.2f} GB, embedding / head {nbytes['grouped'] / 1e9:.2f} GB, other float "
          f"{nbytes['float'] / 1e9:.2f} GB, total {tot / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
