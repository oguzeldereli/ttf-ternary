#!/usr/bin/env python3
"""ttf_qwen: run a TTF-ternarized Qwen3.5 / 3.8 model in its packed form (trits at 1.6 bits per weight, never expanded).

Every large linear of the decoder is a TTF layer,
    y = r * ( beta * (Q(c * x) @ T^T)  +  ((c * x) @ B) @ A^T )
with trits T packed 5 per byte, computed by an int8 x packed-trit GEMM (Triton, bitnet/kernel.py); the rest of the model
(Gated DeltaNet, gated attention, norms, embedding, head) is transformers' own Qwen3.5 code. Needs transformers >= 5.19
and flash-linear-attention.

  python ttf_qwen.py MODEL_DIR WEIGHTS.pt [-p "prompt"] [-i] [--chat] [-n 256] [--temp 0.7] [--emb_cpu]
MODEL_DIR holds config.json and the tokenizer files; WEIGHTS.pt the trainer's model state (scripts: slim27.py).
"""
import argparse, os, sys, time, math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from bitnet.kernel import tern_gemm_i8, trit_beta                    # noqa: E402

_KEEP_FLOAT = ("in_proj_a", "in_proj_b")


class PackedTLinear(nn.Module):
    """inference-only TTF layer on packed trits"""

    def __init__(self, K, N, rank=16):
        super().__init__()
        self.K, self.N = K, N
        self.register_buffer("wpacked", torch.zeros(N, (K + 4) // 5, dtype=torch.uint8))
        self.register_buffer("row_scale", torch.ones(N, dtype=torch.float16))
        self.register_buffer("col_scale", torch.ones(K, dtype=torch.float16))
        self.register_buffer("mag_A", torch.zeros(N, rank, dtype=torch.bfloat16))
        self.register_buffer("mag_B", torch.zeros(K, rank, dtype=torch.bfloat16))
        self.beta = None

    def forward(self, x):
        if self.beta is None:
            self.beta = float(trit_beta(self.wpacked, self.K))
        shp = x.shape
        x = x.reshape(-1, self.K) * self.col_scale.to(x.dtype)
        amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-5).float()
        s = amax / 127.0
        q = (x.float() / s).round().clamp_(-128, 127).to(torch.int8)
        y = tern_gemm_i8(q, s.squeeze(-1), self.wpacked, self.K, self.beta)             # bf16 [M, N]
        y = y + (x.to(torch.bfloat16) @ self.mag_B) @ self.mag_A.t()
        y = y * self.row_scale.to(y.dtype)
        return y.view(*shp[:-1], self.N)


class CPUEmbedding(nn.Module):
    """the 248k x d embedding kept in host memory: a lookup, copied to the GPU per call"""

    def __init__(self, weight, dev):
        super().__init__()
        self.weight_cpu, self.dev = weight, dev

    def forward(self, ids):
        return self.weight_cpu[ids.cpu()].to(self.dev, non_blocking=True)


def build(model_dir, weights, dev, emb_cpu):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    c = AutoConfig.from_pretrained(model_dir)
    t = c.text_config if hasattr(c, "text_config") else c
    t._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = Qwen3_5ForCausalLM(t)
    shapes = {}
    for name, mod in list(hf.model.layers.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname not in _KEEP_FLOAT:
                shapes[(name, cname)] = (child.in_features, child.out_features)
                setattr(mod, cname, nn.Identity())
    emb_shape = tuple(hf.model.embed_tokens.weight.shape)
    if emb_cpu:
        hf.model.embed_tokens = nn.Identity()
    hf = hf.to_empty(device=dev)
    for p in hf.parameters():
        p.data = p.data.to(torch.bfloat16)
    for (name, cname), (i, o) in shapes.items():
        mod = hf.model.layers.get_submodule(name) if name else hf.model.layers
        setattr(mod, cname, PackedTLinear(i, o).to(dev))
    hf.model.rotary_emb = type(hf.model.rotary_emb)(t).to(dev)
    t0 = time.time()
    sd = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("model", sd)
    sd = {k[3:]: v for k, v in sd.items() if k.startswith("hf.")}           # the trainer's QwenTern prefix
    if emb_cpu:
        hf.model.embed_tokens = CPUEmbedding(sd.pop("model.embed_tokens.weight").to(torch.bfloat16).clone(), dev)
    own = hf.state_dict()
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    for k, v in own.items():
        if k in sd:
            v.copy_(sd[k].to(v.dtype))
    print(f"loaded {weights} in {time.time() - t0:.0f}s; missing {len(missing)} {missing[:4]}, unexpected "
          f"{len(unexpected)} {unexpected[:4]}", file=sys.stderr)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    return hf, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir"); ap.add_argument("weights")
    ap.add_argument("-p", "--prompt", default=""); ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
    ap.add_argument("--think", action="store_true", help="with --chat: allow the model's thinking block")
    ap.add_argument("-n", type=int, default=256); ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9); ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--rep", type=float, default=1.05, help="repetition penalty")
    ap.add_argument("--emb_cpu", action="store_true", help="keep the input embedding in host memory (saves 2.4 GiB)")
    ap.add_argument("--ppl", default="", help="uint32 token file: mean loss on the first --n_win windows of 2048")
    ap.add_argument("--n_win", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda"
    hf, t = build(a.model_dir, a.weights, dev, a.emb_cpu)
    print(f"GPU memory {torch.cuda.memory_allocated() / 2**30:.2f} GiB after loading", file=sys.stderr)
    if a.ppl:
        import numpy as np, torch.nn.functional as F
        data = np.memmap(a.ppl, dtype=np.uint32, mode="r")
        ls = []
        with torch.no_grad():
            for i in range(a.n_win):
                x = torch.tensor(data[i * 2048:(i + 1) * 2048 + 1].astype(np.int64), device=dev)[None]
                lg = hf(input_ids=x[:, :-1]).logits
                ls.append(float(F.cross_entropy(lg[0].float(), x[0, 1:])))
        print(f"mean loss {np.mean(ls):.4f} over {len(ls)} windows: " + " ".join(f"{v:.4f}" for v in ls))
        return
    from transformers import AutoTokenizer, TextStreamer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def gen(q):
        if a.chat:
            kw = {} if a.think else {"enable_thinking": False}
            try:
                text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                               add_generation_prompt=True, **kw)
            except TypeError:
                text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                               add_generation_prompt=True)
        else:
            text = q
            print(q, end="", flush=True)
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        t1 = time.time()
        with torch.no_grad():
            out = hf.generate(ids, max_new_tokens=a.n, do_sample=a.temp > 0, temperature=max(a.temp, 1e-5),
                              top_p=a.top_p, top_k=a.top_k, repetition_penalty=a.rep, streamer=streamer,
                              pad_token_id=tok.eos_token_id)
        n = out.shape[1] - ids.shape[1]
        print(f"[{ids.shape[1]} prompt + {n} new tokens, {n / (time.time() - t1):.2f} tok/s, peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB]", file=sys.stderr)

    if a.prompt:
        gen(a.prompt)
    if a.interactive or not a.prompt:
        print("interactive: type a prompt, empty line to quit", file=sys.stderr)
        while True:
            try:
                q = input("\n> ")
            except EOFError:
                break
            if not q.strip():
                break
            gen(q)


if __name__ == "__main__":
    main()
