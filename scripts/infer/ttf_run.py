#!/usr/bin/env python3
"""ttf_run: run ternary (TTF-trained) language models. One file, PyTorch + `tokenizers` only.

A TTF layer computes  y = r * ( beta * (Q(c * x) @ T^T)  +  ((c * x) @ B) @ A^T )
  T      trits {-1, 0, 1}, packed 5 per byte (base 3)
  beta   1 / sqrt(K * density of nonzero trits)  (from the trits)
  c, r   learned per-column / per-row scales;  A, B  a rank-16 float adapter
  Q      per-token absmax int8 quantisation of the activations
inside a Llama-style decoder (RMSNorm, RoPE, SwiGLU, per-head attention temperature, tied embedding / head).

  python ttf_run.py run    MODEL [-p "prompt"] [-i] [-n 128] [--temp 0.8] [--top_p 0.95] [--device cpu]
  python ttf_run.py ppl    MODEL --data val.bin [--n 32] [--seq 2048]       (mean loss on the first N windows)
  python ttf_run.py export CKPT OUT.safetensors                             (trits + scales only, no optimizer state)
MODEL is a trainer checkpoint (.pt) or an exported .safetensors file.
"""
import argparse, json, math, os, sys, time, types

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------- loading
_SKIP = ("frozen_scale",)          # trainer-only buffers


def _stub_unpickle():
    """trainer checkpoints pickle bitnet.config.ModelConfig: make it loadable without the training code"""
    try:
        import bitnet.config  # noqa: F401
    except Exception:
        cfg = types.ModuleType("bitnet.config")

        class ModelConfig:      # attributes restored from the pickle
            pass
        cfg.ModelConfig = ModelConfig
        sys.modules.setdefault("bitnet", types.ModuleType("bitnet"))
        sys.modules["bitnet.config"] = cfg


def load_any(path):
    """-> (config dict, state dict of tensors)"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            cfg = json.loads(f.metadata()["ttf_config"])
        return cfg, load_file(path)
    _stub_unpickle()
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if ck.get("mode", "kernel") != "kernel":
        raise SystemExit(f"{path}: mode {ck.get('mode')} (only TTF / kernel checkpoints carry packed trits)")
    c = ck["cfg"]
    cfg = {k: getattr(c, k) for k in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
                                      "max_seq_len", "rope_theta", "norm_eps", "tie_embeddings", "act_bits")}
    cfg["beta"] = bool(ck.get("beta", True))
    sd = {k: v for k, v in ck["model"].items() if not k.endswith(_SKIP)}
    return cfg, sd


def export(ckpt, out):
    from safetensors.torch import save_file
    cfg, sd = load_any(ckpt)
    bad = [k for k in sd if k.endswith(("bpacked",)) or ".mag_At." in k or "router" in k or "tok_emb.wpacked" in k
           or "tok_emb.table" in k]
    if bad:
        raise SystemExit(f"not supported yet (boost / ternary adapters / MoE / ternary embedding): {bad[:3]}")
    sd = {k: v.contiguous().clone() for k, v in sd.items()}
    save_file(sd, out, metadata={"ttf_config": json.dumps(cfg), "format": "ttf-1"})
    nb = sum(v.numel() * v.element_size() for v in sd.values())
    nt = sum(v.numel() for k, v in sd.items() if k.endswith("wpacked"))
    print(f"wrote {out}: {nb / 2**20:.1f} MiB ({nt / 2**20:.1f} MiB packed trits)")


# ------------------------------------------------------------------------------------------------- model
def unpack_rows(packed, K):
    """uint8 [N, ceil(K/5)] base 3 (digit j has weight 3^j) -> int8 {-1, 0, 1} [N, K]"""
    N, K5 = packed.shape
    x = packed.to(torch.int16)
    out = torch.empty(N, K5, 5, dtype=torch.int8)
    for j in range(5):
        out[:, :, j] = (x % 3).to(torch.int8)
        x = x // 3
    return out.view(N, K5 * 5)[:, :K] - 1


class TLinear:
    def __init__(self, sd, pre, K, dt, dev, use_beta=True, act_bits=8):
        T = unpack_rows(sd[pre + "wpacked"], K)
        self.N, self.K = T.shape
        self.beta = float((T != 0).sum()) / self.N
        self.beta = 1.0 / math.sqrt(max(self.beta, 1.0)) if use_beta else 1.0
        self.W = T.to(dt).to(dev)                          # exact: trits are small integers
        g = lambda k: sd[pre + k].to(dt).to(dev) if (pre + k) in sd else None
        self.r, self.c, self.A, self.B = g("row_scale"), g("col_scale"), g("mag_A"), g("mag_B")
        self.qp = 2 ** (act_bits - 1) - 1

    def __call__(self, x):
        if self.c is not None:
            x = x * self.c
        amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-5)
        s = amax / self.qp
        q = (x / s).round().clamp_(-self.qp - 1, self.qp)
        y = (q @ self.W.T) * (s * self.beta)
        if self.A is not None:
            y = y + (x @ self.B) @ self.A.T
        if self.r is not None:
            y = y * self.r
        return y


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class TTFModel:
    def __init__(self, cfg, sd, dev, dt):
        self.cfg, self.dev, self.dt = cfg, dev, dt
        d, H, Hkv = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
        self.H, self.Hkv, self.hd, self.eps = H, Hkv, d // H, cfg["norm_eps"]
        g = lambda k: sd[k].to(dt).to(dev)
        self.emb = g("tok_emb.weight")
        self.head = self.emb if cfg.get("tie_embeddings", True) or "lm_head.weight" not in sd else g("lm_head.weight")
        self.norm = g("norm.weight")
        ub, ab = cfg.get("beta", True), cfg.get("act_bits", 8)
        self.layers = []
        for i in range(cfg["n_layers"]):
            p = f"layers.{i}."
            L = types.SimpleNamespace()
            L.n1, L.n2 = g(p + "attn_norm.weight"), g(p + "ffn_norm.weight")
            L.temp = g(p + "attn.qk_logscale").exp() if (p + "attn.qk_logscale") in sd else None
            L.wq, L.wk, L.wv = (TLinear(sd, p + f"attn.{n}.", d, dt, dev, ub, ab) for n in ("wq", "wk", "wv"))
            L.wo = TLinear(sd, p + "attn.wo.", H * self.hd, dt, dev, ub, ab)
            L.wg, L.wu = (TLinear(sd, p + f"ffn.{n}.", d, dt, dev, ub, ab) for n in ("w_gate", "w_up"))
            L.wd = TLinear(sd, p + "ffn.w_down.", cfg["hidden_dim"], dt, dev, ub, ab)
            self.layers.append(L)
        hd = self.hd
        fr = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2)[: hd // 2].float() / hd))
        t = torch.arange(cfg["max_seq_len"])
        self.rope = torch.polar(torch.ones(len(t), hd // 2), torch.outer(t, fr)).to(dev)

    def _rot(self, x, pos):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = self.rope[pos:pos + x.shape[1]].unsqueeze(0).unsqueeze(2)
        return torch.view_as_real(xc * f).flatten(-2).to(x.dtype)

    @torch.no_grad()
    def forward(self, idx, cache=None, pos=0):
        """idx [B, T] -> logits [B, T, V]; cache: list of (k, v) per layer, extended in place"""
        B, T = idx.shape
        h = self.emb[idx]
        for li, L in enumerate(self.layers):
            x = rmsnorm(h, L.n1, self.eps)
            q = L.wq(x).view(B, T, self.H, self.hd)
            k = L.wk(x).view(B, T, self.Hkv, self.hd)
            v = L.wv(x).view(B, T, self.Hkv, self.hd)
            if L.temp is not None:
                q = q * L.temp.view(1, 1, -1, 1)
            q, k = self._rot(q, pos), self._rot(k, pos)
            if cache is not None:
                if len(cache) > li:
                    k = torch.cat([cache[li][0], k], 1); v = torch.cat([cache[li][1], v], 1)
                    cache[li] = (k, v)
                else:
                    cache.append((k, v))
            rep = self.H // self.Hkv
            kk, vv = (t.repeat_interleave(rep, 2) if rep > 1 else t for t in (k, v))
            o = F.scaled_dot_product_attention(q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
                                               is_causal=(T > 1 and pos == 0))
            h = h + L.wo(o.transpose(1, 2).reshape(B, T, -1))
            x = rmsnorm(h, L.n2, self.eps)
            h = h + L.wd(F.silu(L.wg(x)) * L.wu(x))
        return rmsnorm(h, self.norm, self.eps) @ self.head.T


# ------------------------------------------------------------------------------------------------- sampling
def sample(logits, temp, top_k, top_p):
    if temp <= 0:
        return int(logits.argmax())
    logits = logits.float() / temp
    if top_k:
        v, _ = logits.topk(top_k)
        logits[logits < v[-1]] = -float("inf")
    p = logits.softmax(-1)
    if top_p < 1.0:
        sp, si = p.sort(descending=True)
        keep = sp.cumsum(0) - sp < top_p
        p = torch.zeros_like(p).scatter_(0, si[keep], sp[keep])
    return int(torch.multinomial(p / p.sum(), 1))


def generate(model, tok, prompt, n, temp, top_k, top_p, bos=True, stream=True):
    eos = tok.token_to_id("</s>")
    ids = ([eos] if bos else []) + tok.encode(prompt, add_special_tokens=False).ids
    ctx = model.cfg["max_seq_len"]
    ids = ids[-ctx:]
    cache, t0 = [], time.time()
    logits = model.forward(torch.tensor([ids], device=model.dev), cache, 0)[0, -1]
    t1 = time.time()
    out = []
    pids = [t for t in ids if t != eos]
    base = tok.decode(pids)            # decode prompt + continuation together: sentencepiece drops a leading space
    shown = base
    for i in range(n):
        nxt = sample(logits, temp, top_k, top_p)
        if nxt == eos:
            break
        out.append(nxt)
        if stream:
            txt = tok.decode(pids + out)
            sys.stdout.write(txt[len(shown):]); sys.stdout.flush(); shown = txt
        if len(ids) + len(out) >= ctx:
            break
        logits = model.forward(torch.tensor([[nxt]], device=model.dev), cache, len(ids) + len(out) - 1)[0, -1]
    t2 = time.time()
    if stream:
        print(f"\n[prompt {len(ids)} tok in {t1 - t0:.2f}s | {len(out)} tok, {len(out) / max(t2 - t1, 1e-9):.1f} tok/s]")
    return tok.decode(pids + out)[len(base):]


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "ppl", "export"]); ap.add_argument("model"); ap.add_argument("out", nargs="?")
    ap.add_argument("-p", "--prompt", default=""); ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("-n", type=int, default=128); ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40); ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--dtype", default="", help="float32 (CPU default) / bfloat16 (CUDA default); not float16: the integer sums of int8 x trits overflow it")
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="hf-internal-testing/llama-tokenizer")
    ap.add_argument("--no_bos", action="store_true", help="do not start the prompt with </s> (document boundary)")
    ap.add_argument("--data", default=""); ap.add_argument("--n_win", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    a = ap.parse_args()
    if a.cmd == "export":
        return export(a.model, a.out)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.seed:
        torch.manual_seed(a.seed)
    dt = getattr(torch, a.dtype) if a.dtype else (torch.float32 if a.device == "cpu" else torch.bfloat16)
    t0 = time.time()
    cfg, sd = load_any(a.model)
    model = TTFModel(cfg, sd, a.device, dt)
    del sd
    print(f"{a.model}: {cfg['n_layers']} layers, width {cfg['dim']}, vocab {cfg['vocab_size']}; loaded in "
          f"{time.time() - t0:.1f}s on {a.device} ({str(dt).replace('torch.', '')})", file=sys.stderr)
    if a.cmd == "ppl":
        import numpy as np
        data = np.memmap(a.data, dtype=np.uint16, mode="r")
        ls = []
        for i in range(a.n_win):
            x = torch.tensor(data[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64), device=a.device)[None]
            lg = model.forward(x[:, :-1])
            ls.append(float(F.cross_entropy(lg[0].float(), x[0, 1:])))
        print(f"mean loss {np.mean(ls):.4f} over {a.n_win} windows of {a.seq} (ppl {math.exp(np.mean(ls)):.2f})")
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(a.tokenizer)
    if a.prompt:
        print(a.prompt, end="")
        generate(model, tok, a.prompt, a.n, a.temp, a.top_k, a.top_p, not a.no_bos)
    if a.interactive or not a.prompt:
        print("interactive: type a prompt, empty line to quit", file=sys.stderr)
        while True:
            try:
                q = input("\n> ")
            except EOFError:
                break
            if not q.strip():
                break
            generate(model, tok, q, a.n, a.temp, a.top_k, a.top_p, not a.no_bos)


if __name__ == "__main__":
    main()
