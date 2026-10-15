"""Shared loaders for the 110M analyses: any of our checkpoints (fp32 / kernel / master), and a GPTQ rounding of an fp32
checkpoint (dequantised; input second moments from 32 random training windows)."""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from bitnet.model import BitTransformer                                          # noqa: E402
from bitnet.hf_init import ternarize, ternarize_gptq                              # noqa: E402

LIN = [("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("attn", "wo"), ("ffn", "w_gate"), ("ffn", "w_up"), ("ffn", "w_down")]


def load(path, dev="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if ck["mode"] == "fp32":
        m = BitTransformer(ck["cfg"], grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev)
        m.load_state_dict(ck["model"], strict=True)
        return m.eval()
    from final_eval import load_model
    return load_model(path, dev)[0].eval()


@torch.no_grad()
def gptq_copy(path, train="data/wiki32k_train.bin", seq=2048, n=32, dev="cuda"):
    m = load(path, dev)
    H, cnt, hooks = {}, {}, []
    for i, blk in enumerate(m.layers):
        for sub, nm in LIN:
            lin = getattr(getattr(blk, sub), nm)
            H[(i, nm)] = torch.zeros(lin.in_features, lin.in_features, device=dev); cnt[(i, nm)] = 0

            def hk(mod, inp, key=(i, nm)):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H[key].addmm_(x.T, x); cnt[key] += x.shape[0]
            hooks.append(lin.register_forward_pre_hook(hk))
    tr = np.memmap(train, dtype=np.uint16, mode="r")
    for s in np.random.default_rng(4321).integers(0, len(tr) - seq - 1, n):
        x = torch.tensor(tr[s:s + seq + 1].astype(np.int64), device=dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    for i, blk in enumerate(m.layers):
        for sub, nm in LIN:
            lin = getattr(getattr(blk, sub), nm); W = lin.weight.data.float()
            T, ra, c = ternarize(W, "als"); T, ra, c = ternarize_gptq(W, H[(i, nm)] / cnt[(i, nm)], ra, c)
            lin.weight.data = ra[:, None] * c[None, :] * T.float()
    del H
    return m
