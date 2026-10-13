#!/usr/bin/env python3
"""ttf_qwen: run a TTF-ternarized Qwen3.5 / 3.8 model in its packed form (trits at 1.6 bits per weight, never expanded).

Every large linear of the decoder is a TTF layer,
    y = r * ( beta * (Q(c * x) @ T^T)  +  ((c * x) @ B) @ A^T )
with trits T packed 5 per byte, computed by an int8 x packed-trit GEMM (Triton, bitnet/kernel.py); the rest of the model
(Gated DeltaNet, gated attention, norms, embedding, head) is transformers' own Qwen3.5 code. Needs transformers >= 5.19
and flash-linear-attention.

  python ttf_qwen.py MODEL_DIR WEIGHTS.pt [-p "prompt"] [-i] [--chat] [-n 256] [--temp 0.7]
MODEL_DIR holds config.json and the tokenizer files; WEIGHTS.pt the trainer's model state (scripts: slim27.py).
"""
import argparse, os, sys, time, math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from bitnet.kernel import tern_gemm_i8, trit_beta                    # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf_kernels import act_quant_planes, tern_gemv_planes, tern_gemv2, repack_2bit, _UVA_TABLES, _gemv2_epi_kernel   # noqa: E402
import triton                                                          # noqa: E402
import ttf_kernels                                                     # noqa: E402,F401  (registers ttf::uva_embed)
DECODE_M = 16      # up to this many tokens per call: the GEMV path (prefill of short prompts too)
TWO_BIT = os.environ.get("TTF_TWO_BIT", "1") == "1"   # 2-bit layout + dp4a (faster, +25% trit memory)

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
        x2 = x.reshape(-1, self.K)
        if getattr(self, "w2", None) is not None and x2.shape[0] <= DECODE_M:   # decode: GEMV with the epilogue fused
            M_, K4p = x2.shape[0], self.w2.shape[1]
            xc = x2 * self.col_scale.to(x2.dtype)
            sc = xc.abs().amax(-1).clamp_min(1e-5).float() / 127.0
            q = (xc.float() / sc[:, None]).round().clamp_(-128, 127).to(torch.int8)
            if self.K < K4p * 4:
                q = torch.nn.functional.pad(q, (0, K4p * 4 - self.K))
            sumx = q.to(torch.int32).sum(1)
            r = (xc.to(torch.bfloat16) @ self.mag_B).float().contiguous()
            xw = q.contiguous().view(torch.int32).view(M_, K4p)
            ww = self.w2.view(torch.int32)
            y = torch.zeros(M_, self.N, dtype=torch.float32, device=x.device)
            grid = lambda meta: (triton.cdiv(self.N, meta["BN"]), M_, meta["SPLIT"])
            _gemv2_epi_kernel[grid](xw, ww, y, sc, sumx, r, self.mag_A, self.row_scale, self.beta, self.N, K4p // 4,
                                    M_, self.mag_A.shape[1], xw.stride(0), ww.stride(0), y.stride(0),
                                    BR=triton.next_power_of_2(max(self.mag_A.shape[1], 2)))
            return y.to(torch.bfloat16).view(*shp[:-1], self.N)
        if getattr(self, "w2", None) is not None:      # 2-bit layout (repacked at load): dp4a GEMV / per-layer expand
            xc = x2 * self.col_scale.to(x2.dtype)
            s = xc.abs().amax(-1, keepdim=True).clamp_min(1e-5).float() / 127.0
            q = (xc.float() / s).round().clamp_(-128, 127).to(torch.int8)
            if x2.shape[0] <= DECODE_M:
                acc = tern_gemv2(q, self.w2).float()
            else:
                sh = torch.tensor([0, 2, 4, 6], device=x.device, dtype=torch.uint8)
                Wt = ((self.w2[:, :, None] >> sh) & 3).reshape(self.N, -1)[:, :self.K].to(torch.bfloat16) - 1
                acc = (q.to(torch.bfloat16) @ Wt.t()).float()
                del Wt
            y = (acc * (s * self.beta)).to(torch.bfloat16) + (xc.to(torch.bfloat16) @ self.mag_B) @ self.mag_A.t()
            return (y * self.row_scale.to(y.dtype)).view(*shp[:-1], self.N)
        if x2.shape[0] <= DECODE_M:     # decode: fused quantisation + a GEMV that reads the packed trits once
            xp, sc = act_quant_planes(x2.contiguous(), self.col_scale, self.wpacked.shape[1])
            y = (tern_gemv_planes(xp, self.wpacked).float() * (sc[:, None] * self.beta)).to(torch.bfloat16)
            x = x2 * self.col_scale.to(x2.dtype)
        else:
            x = x2 * self.col_scale.to(x.dtype)
            amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-5).float()
            s = amax / 127.0
            q = (x.float() / s).round().clamp_(-128, 127).to(torch.int8)
            y = tern_gemm_i8(q, s.squeeze(-1), self.wpacked, self.K, self.beta)         # bf16 [M, N]
        y = y + (x.to(torch.bfloat16) @ self.mag_B) @ self.mag_A.t()
        y = y * self.row_scale.to(y.dtype)
        return y.view(*shp[:-1], self.N)


class UVAEmbedding(nn.Module):
    """the 248k x d embedding in pinned host memory, read by a GPU kernel through unified addressing: no host sync and
    no graph break (the lookup is a custom op), 2.4 GiB of GPU memory saved; exact (the same bf16 rows)"""

    def __init__(self, weight, dev):
        super().__init__()
        self.tab = weight.contiguous().pin_memory()
        self.key = id(self)
        _UVA_TABLES[self.key] = self.tab
        self.dim = weight.shape[1]

    def forward(self, ids):
        return torch.ops.ttf.uva_embed(ids, self.key, self.dim)


class CPUEmbedding(nn.Module):
    """the 248k x d embedding kept in host memory: a lookup, copied to the GPU per call"""

    def __init__(self, weight, dev):
        super().__init__()
        self.weight_cpu, self.dev = weight, dev

    @torch.compiler.disable             # a host lookup: kept out of the compiled / CUDA-graph region
    def forward(self, ids):
        return self.weight_cpu[ids.cpu()].to(self.dev)


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
    hf = hf.to(torch.bfloat16)            # still on meta: free; to_empty then allocates bf16 directly (no fp32 copy)
    hf = hf.to_empty(device=dev)
    for (name, cname), (i, o) in shapes.items():
        mod = hf.model.layers.get_submodule(name) if name else hf.model.layers
        setattr(mod, cname, PackedTLinear(i, o).to(dev))
    hf.model.rotary_emb = type(hf.model.rotary_emb)(t).to(dev)
    t0 = time.time()
    sd = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("model", sd)
    sd = {k[3:]: v for k, v in sd.items() if k.startswith("hf.")}           # the trainer's QwenTern prefix
    if emb_cpu:
        hf.model.embed_tokens = UVAEmbedding(sd.pop("model.embed_tokens.weight").to(torch.bfloat16), dev)
    own = hf.state_dict()
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    for k, v in own.items():
        if k in sd:
            v.copy_(sd[k].to(v.dtype))
    del own, sd                         # the state dict holds the unpadded buffers: release them first
    torch.cuda.empty_cache()
    for mod in hf.modules():            # rows padded to 16 bytes (aligned, vectorised loads); the padding byte 121
        if isinstance(mod, PackedTLinear):  # decodes to five zero trits; wpacked stays the logical [N, K5] view
            N_, K5_ = mod.wpacked.shape
            K5p = (K5_ + 15) // 16 * 16
            if K5p != K5_:
                buf = torch.full((N_, K5p), 121, dtype=torch.uint8, device=dev)
                buf[:, :K5_] = mod.wpacked
                mod.wpacked = buf[:, :K5_]
            mod.beta = float(trit_beta(mod.wpacked, mod.K))   # at load: no host sync inside a CUDA graph
            if TWO_BIT:                     # 2-bit layout for the dp4a decode kernel; the base-3 buffer is dropped
                mod.w2 = repack_2bit(mod.wpacked, mod.K)
                mod.wpacked = torch.empty(0, dtype=torch.uint8, device=dev)
                torch.cuda.empty_cache()
    print(f"loaded {weights} in {time.time() - t0:.0f}s; missing {len(missing)} {missing[:4]}, unexpected "
          f"{len(unexpected)} {unexpected[:4]}", file=sys.stderr)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    return hf, t


def fast_generate(hf, t, ids, n, temp=0.0, top_p=1.0, top_k=0, eos=None, max_len=4096, on_tokens=None, every=8,
                  eos_set=None):
    """decode with a static cache and a compiled (CUDA-graph) forward; the next token stays on the GPU, the host syncs
    every `every` tokens (to stream them and check for eos). Greedy for temp 0, else temperature / top-k / top-p."""
    from transformers import StaticCache
    dev = ids.device
    if not hasattr(hf, "_ttf_step"):
        hf._ttf_step = torch.compile(hf.forward, mode="reduce-overhead", fullgraph=False)
    cache = getattr(hf, "_ttf_cache", None)       # one cache, reset per prompt: the compiled step is reused (no recompile)
    if cache is None or getattr(hf, "_ttf_cache_len", 0) != max_len:
        cache = hf._ttf_cache = StaticCache(config=t, max_cache_len=max_len)
        hf._ttf_cache_len = max_len
    else:
        cache.reset()
    L = ids.shape[1]
    _t0 = time.time()
    with torch.no_grad():
        logits = hf(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device=dev),
                    use_cache=True).logits[:, -1].float()

        def pick(lg):
            if temp <= 0:
                return lg.argmax(-1, keepdim=True)
            lg = lg / temp
            if top_k:
                v, _ = lg.topk(top_k)
                lg = lg.masked_fill(lg < v[:, -1:], -float("inf"))
            p = lg.softmax(-1)
            if top_p < 1.0:
                sp, si = p.sort(-1, descending=True)
                keep = sp.cumsum(-1) - sp < top_p
                p = torch.zeros_like(p).scatter_(-1, si, sp * keep)
            return torch.multinomial(p / p.sum(-1, keepdim=True), 1)

        out, pend = [], []
        tok = pick(logits)
        pos = torch.tensor([L], device=dev)
        if on_tokens is None and os.environ.get("TTF_TIMING"):
            torch.cuda.synchronize(); print(f"  prefill {L} tokens: {time.time() - _t0:.2f}s", flush=True)
        for i in range(n):
            pend.append(tok)
            if len(pend) == every or i == n - 1:
                got = torch.cat(pend, 1)[0].tolist(); pend = []
                stops = (eos_set or set()) | ({eos} if eos is not None else set())
                hit = [k for k, g in enumerate(got) if g in stops]
                stop = bool(hit)
                if stop:
                    got = got[:hit[0]]
                out += got
                if on_tokens:
                    on_tokens(got)
                if stop:
                    break
            if i == n - 1 or L + i + 1 >= max_len:
                break
            logits = hf._ttf_step(input_ids=tok, past_key_values=cache, cache_position=pos, use_cache=True).logits[:, -1].float()
            tok = pick(logits)
            pos = pos + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir"); ap.add_argument("weights")
    ap.add_argument("-p", "--prompt", default=""); ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
    ap.add_argument("--think", action="store_true", help="with --chat: allow the model's thinking block")
    ap.add_argument("-n", type=int, default=256); ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9); ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--rep", type=float, default=1.05, help="repetition penalty")
    ap.add_argument("--emb_gpu", action="store_true", help="keep the input embedding on the GPU (default: pinned host "
                    "memory read by the GPU, exact and 2.4 GiB smaller)")
    ap.add_argument("--slow", action="store_true", help="transformers' generate instead of the compiled decode loop")
    ap.add_argument("--ctx", type=int, default=4096, help="static cache length (prompt + new tokens)")
    ap.add_argument("--ppl", default="", help="uint32 token file: mean loss on the first --n_win windows of 2048")
    ap.add_argument("--n_win", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda"
    hf, t = build(a.model_dir, a.weights, dev, not a.emb_gpu)
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
        if a.slow:
            with torch.no_grad():
                out = hf.generate(ids, max_new_tokens=a.n, do_sample=a.temp > 0, temperature=max(a.temp, 1e-5),
                                  top_p=a.top_p, top_k=a.top_k, repetition_penalty=a.rep, streamer=streamer,
                                  pad_token_id=tok.eos_token_id)
            n = out.shape[1] - ids.shape[1]
        else:
            shown = {"ids": [], "txt": ""}

            def emit(got):
                shown["ids"] += got
                txt = tok.decode(shown["ids"], skip_special_tokens=True)
                sys.stdout.write(txt[len(shown["txt"]):]); sys.stdout.flush(); shown["txt"] = txt
            eos = [tok.eos_token_id] + [tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|endoftext|>")]
            out = fast_generate(hf, t, ids, a.n, temp=a.temp, top_p=a.top_p, top_k=a.top_k, eos=None,
                                max_len=a.ctx, on_tokens=emit, eos_set=set(e for e in eos if e is not None))
            n = len(out); print()
        print(f"[{ids.shape[1]} prompt + {n} new tokens, {n / (time.time() - t1):.2f} tok/s incl. prefill, peak "
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
