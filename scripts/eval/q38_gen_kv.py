"""q38_gen_kv.py: GSM8K / HumanEval generation for a Q38 slim checkpoint on a 24 GB lab GPU with
incremental decoding (q38_eval_lab.py recomputes the whole sequence per token: 25 s / step at batch 64).
- Prefill: one sequence at a time at its exact length (no padding: the ternary projections' biases / centring would
  leak padded positions into DeltaNet's state), but layer by layer, so each layer moves host -> GPU once per batch.
- Decode: all rows of the batch together, one token per step. State per layer:
  * attention (12 layers): K / V in a [B, kvh, Lmax, hd] buffer, each row written at its own position, a key mask;
  * DeltaNet (36): the reference's conv state (last 4 pre-conv inputs) and recurrent state (fp32), stacked over rows;
  * PLE (layer 1): the last (k - 1) x n norm_conv(gv) rows (its dilated causal conv), the n-gram from the last n ids.
- Rows that hit EOS keep running (outputs ignored) until the batch ends: no index juggling.
Prompts, template and greedy decoding as gen_bench / q38_eval_lab. --check N: N decode steps against q38_eval_lab's
full-recompute forward (argmax and logit difference), then exit.

  python scripts/eval/q38_gen_kv.py --model BASE --slim DIR --bench DIR --out DIR [--tasks gsm8k:200,humaneval:164] [--bs 64]
         [--max-new 512] [--resident 0] [--layers 0] [--check 0]
  fp teacher : --fp EXPDIR --fp-stack DIR [--fp-pin N] in place
  of --slim (experts FP8 on the GPU, dequantised per routed expert; N layers' experts pinned on the host, the rest
  streamed from per-layer stack files)"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38.load import load_slim
from q38_eval_lab import items
from ttf.q38.train import tern_modules



class _Slot:
    def __init__(self):
        self.record_past = False
        self.conv_states = [None]
        self.recurrent_states = [None]


class LinCache:
    """the cache interface the reference's Qwen3_5MoeGatedDeltaNet.forward uses, for one layer's rows"""
    def __init__(self, idx, conv=None, rec=None, k=4):
        self.layers = {idx: _Slot()}
        self.k = k
        self.layers[idx].conv_states[0] = conv
        self.layers[idx].recurrent_states[0] = rec

    def has_previous_state(self, idx, state_idx=0):
        return self.layers[idx].recurrent_states[0] is not None

    def update_conv_state(self, x, idx, conv_kernel_size=None, **kw):
        k = conv_kernel_size or self.k
        s = x[..., -k:]
        if s.shape[-1] < k:
            s = F.pad(s, (k - s.shape[-1], 0))
        self.layers[idx].conv_states[0] = s.contiguous()
        return x

    def update_recurrent_state(self, s, idx, **kw):
        self.layers[idx].recurrent_states[0] = s
        return s


class KV:
    """attention K / V for the batch: buf [B, kvh, Lmax, hd]; prefill (row r): written at 0..T-1, returned as is;
    decode: each row's new key at pos[r], returned up to max(pos) + 1 (the mask hides the rest)"""
    def __init__(self, B, kvh, Lmax, hd, dev, dtype):
        self.k = torch.zeros(B, kvh, Lmax, hd, device=dev, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.row, self.pos, self.lim = None, None, None

    def update(self, k, v, idx, **kw):
        if self.row is not None:                       # prefill of one row
            T = k.shape[2]
            self.k[self.row, :, :T] = k[0]; self.v[self.row, :, :T] = v[0]
            return k, v
        b = torch.arange(k.shape[0], device=k.device)
        self.k[b, :, self.pos] = k[:, :, 0]; self.v[b, :, self.pos] = v[:, :, 0]
        return self.k[:, :, :self.lim], self.v[:, :, :self.lim]


def ple_fwd(P, h, ids, state):
    """the PLE block (ttf.q38.model.PLE.forward) for the last h.shape[1] positions of ids, with the conv state
    [B, D, pad] (zeros at the start); returns (out, new state)"""
    T = h.shape[1]
    e = P.ple_embedding(ids)[:, -T:].to(h.dtype)
    k = P.norm_key(P.key_proj(e)).unflatten(-1, (P.hc, P.d))
    v = P.value_proj(e)
    q = P.norm_query(h).unflatten(-1, (P.hc, P.d))
    g = (k * q).sum(-1, keepdim=True) / math.sqrt(P.d)
    g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
    gv = (torch.sigmoid(g) * v.unsqueeze(-2)).flatten(-2)
    c = torch.cat([state, P.norm_conv(gv).transpose(1, 2)], -1)
    out = gv + F.silu(P.conv1d(c)).transpose(1, 2)
    return out, c[..., -P.pad:].contiguous()


def layer_step(L, i, h, pe, ple_ids, st, mask):
    """one layer on h with its state st (dict); prefill (one row) or decode (all rows)"""
    if L.ple is not None:
        o, st["ple"] = ple_fwd(L.ple, h, ple_ids, st["ple"])
        h = h + o
    x, w = L.attn_hyper_connection(h)
    if L.kind == "linear_attention":
        y = L.linear_attn(x, cache_params=st["lin"])
    else:
        y, _ = L.self_attn(x, pe, mask, past_key_values=st["kv"])
    h = QM.GatedResidual.inject(h, y, w)
    x, w = L.mlp_hyper_connection(h)
    return QM.GatedResidual.inject(h, L.mlp(x), w)


class Streamer:
    """the non-resident layers' tensors in pinned host memory, copied to the GPU on a side stream (the next layer's copy
    overlaps this layer's compute); same placement semantics as L.to(dev) / L.to("cpu") (parameters and buffers)"""
    def __init__(self, m, a, dev):
        self.m, self.a, self.dev = m, a, dev
        self.cpu, self.pend = {}, {}
        self.side = torch.cuda.Stream()
        for i, L in enumerate(m.layers):
            if i < a.resident:
                continue
            ts = []
            for mod in L.modules():
                for d in (mod._parameters, mod._buffers):
                    for k, t in d.items():
                        if t is None or t.device.type != "cpu":
                            continue
                        pin = t.data.pin_memory()
                        t.data = pin
                        ts.append((t, pin))
            self.cpu[i] = ts
        print(f"streamer: {len(self.cpu)} layers pinned, "
              f"{sum(p.numel() * p.element_size() for ts in self.cpu.values() for _, p in ts) / 2**30:.1f} GiB", flush=True)

    def prefetch(self, i):
        if i not in self.cpu or i in self.pend:
            return
        with torch.cuda.stream(self.side):
            g = [p.to(self.dev, non_blocking=True) for _, p in self.cpu[i]]
            e = torch.cuda.Event(); e.record(self.side)
        self.pend[i] = (g, e)

    def on(self, i):
        L = self.m.layers[i]
        if i in self.cpu:
            self.prefetch(i)
            g, e = self.pend.pop(i)
            cur = torch.cuda.current_stream()
            cur.wait_event(e)
            for (t, _), x in zip(self.cpu[i], g):
                x.record_stream(cur)
                t.data = x
        self.prefetch((i + 1) % len(self.m.layers))
        for _, t in tern_modules(L):
            t.invalidate()

    def off(self, i):
        L = self.m.layers[i]
        for _, t in tern_modules(L):
            t._wc = None
        if i in self.cpu:
            for t, p in self.cpu[i]:
                t.data = p


_ST = [None]
_FP = [None]


def _on(L, i, a, dev):
    _ST[0].on(i)
    if _FP[0] is not None:
        _FP[0].on(i)


def _off(L, i, a):
    _ST[0].off(i)
    if _FP[0] is not None:
        _FP[0].off(i)


# ---- --fp EXPDIR: the fp teacher (the FP8 release: experts FP8 e4m3 + 128 x 128 block scales, everything else bf16)
class FP8Mat(torch.nn.Module):
    """DenseMat with FP8 weights on the GPU (w8 uint8 [E, N, K], s [E, N/128, K/128]), each routed expert dequantised
    on the fly (W = w x s per 128 x 128 tile, as ttf.q38.model.dequant_fp8), so no bf16 copy of all 512 experts"""

    def __init__(self, E, N, K):
        super().__init__()
        self.E, self.N, self.K = E, N, K
        self.w8 = self.s = None
        self.bias = None

    def forward(self, x, offsets):
        out = x.new_empty(x.shape[0], self.N)
        o = offsets.tolist()
        w8 = self.w8.view(torch.float8_e4m3fn)
        for e in range(self.E):
            if o[e + 1] > o[e]:
                s = self.s[e].repeat_interleave(128, 0)[:self.N].repeat_interleave(128, 1)[:, :self.K]
                out[o[e]:o[e + 1]] = F.linear(x[o[e]:o[e + 1]], (w8[e].float() * s).to(x.dtype))
        return out


def fp_stack(expdir, stack, i, E, I, d):
    """layer i's experts from the release shards into one file: gu8 [E, 2I, d] (gate rows, then up), gus, dn8, dns"""
    fn = os.path.join(stack, f"L{i}.pt")
    if os.path.exists(fn):
        return fn
    rd = QM.Reader(expdir)
    u8 = torch.uint8
    out = dict(gu8=torch.empty(E, 2 * I, d, dtype=u8), gus=torch.empty(E, 2 * I // 128, d // 128),
               dn8=torch.empty(E, d, I, dtype=u8), dns=torch.empty(E, d // 128, I // 128))
    for e in range(E):
        q = f"{QM.PFX}layers.{i}.mlp.experts.{e}."
        out["gu8"][e, :I] = rd.get(q + "gate_proj.weight").view(u8)
        out["gu8"][e, I:] = rd.get(q + "up_proj.weight").view(u8)
        out["gus"][e, :I // 128] = rd.get(q + "gate_proj.weight_scale_inv").float()
        out["gus"][e, I // 128:] = rd.get(q + "up_proj.weight_scale_inv").float()
        out["dn8"][e] = rd.get(q + "down_proj.weight").view(u8)
        out["dns"][e] = rd.get(q + "down_proj.weight_scale_inv").float()
    torch.save(out, fn + ".part"); os.rename(fn + ".part", fn)
    print(f"fp stack L{i} written", flush=True)
    return fn


class FPExperts:
    """the routed experts of every layer: the first --fp-pin layers in pinned host memory, the rest read from their
    stack file each time (a background thread loads layer i + 1 into a pinned staging buffer while layer i runs)"""

    def __init__(self, m, a, dev):
        from concurrent.futures import ThreadPoolExecutor
        self.m, self.dev = m, dev
        cfg = m.cfg
        E, I, d = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
        os.makedirs(a.fp_stack, exist_ok=True)
        self.fn = [fp_stack(a.fp, a.fp_stack, i, E, I, d) for i in range(len(m.layers))]
        self.pin = {}
        for i in range(min(a.fp_pin, len(m.layers))):
            b = torch.load(self.fn[i], map_location="cpu", mmap=True, weights_only=True)
            self.pin[i] = {k: v.pin_memory() for k, v in b.items()}
        self.stage = [None, None]
        self.ex = ThreadPoolExecutor(1); self.fut = {}
        print(f"fp experts: {len(self.pin)} layers pinned, {len(m.layers) - len(self.pin)} from disk", flush=True)

    def _host(self, i):
        if i in self.pin:
            return self.pin[i]
        b = torch.load(self.fn[i], map_location="cpu", mmap=True, weights_only=True)
        k_ = i % 2
        if self.stage[k_] is None:
            self.stage[k_] = {k: torch.empty_like(v).pin_memory() for k, v in b.items()}
        for k, v in b.items():
            self.stage[k_][k].copy_(v)
        return self.stage[k_]

    def on(self, i):
        h = self.fut.pop(i).result() if i in self.fut else self._host(i)
        mo = self.m.layers[i].mlp
        mo.gu.w8, mo.gu.s = h["gu8"].to(self.dev), h["gus"].to(self.dev)
        mo.dn.w8, mo.dn.s = h["dn8"].to(self.dev), h["dns"].to(self.dev)
        n = (i + 1) % len(self.m.layers)
        if n not in self.pin and n not in self.fut:
            self.fut[n] = self.ex.submit(self._host, n)

    def off(self, i):
        mo = self.m.layers[i].mlp
        mo.gu.w8 = mo.gu.s = mo.dn.w8 = mo.dn.s = None


def build_fp(a, dev):
    """the fp teacher: q38_eval_lab.build without the slim checkpoint and the ternary swap; routed experts FP8Mat"""
    cfg = QM.text_config(a.model)
    m = QM.build(cfg, layers=a.layers or None, device="meta")
    rd = QM.Reader(a.model)
    QM.load_globals(m, rd, "cpu")
    for i, L in enumerate(m.layers):
        QM.load_layer(L, i, rd, "cpu", experts=False)
        L.mlp.gu = FP8Mat(L.mlp.E, 2 * L.mlp.I, cfg.hidden_size)
        L.mlp.dn = FP8Mat(L.mlp.E, cfg.hidden_size, L.mlp.I)
        L.to("cpu")
        if L.ple is not None:
            L.ple.ple_embedding.table, L.ple.ple_embedding.scale = QM.ngram_table_memmap(a.model, i)
        if i < a.resident:
            L.to(dev)
        print(f"L{i} loaded (fp){' resident' if i < a.resident else ''}", flush=True)
    for mod in (m.embed_tokens, m.lm_head, m.hyper_connection_mixer):
        mod.to(dev)
    m.rotary.to(dev)
    m.eval()
    print(f"fp teacher {a.model} + experts {a.fp}", flush=True)
    return m


class Gen:
    def __init__(self, m, a, dev, eos):
        self.m, self.a, self.dev, self.eos = m, a, dev, eos
        cfg = m.cfg
        self.kvh, self.hd = cfg.num_key_value_heads, cfg.head_dim

    @torch.no_grad()
    def run(self, seqs, max_new, check=0):
        """greedy continuation of each id list in seqs (one batch); returns the generated id lists"""
        m, a, dev = self.m, self.a, self.dev
        B = len(seqs); lens = torch.tensor([len(s) for s in seqs], device=dev)
        Lmax = int(lens.max()) + max_new + 1
        pl = [L for L in m.layers if L.ple is not None]
        n = pl[0].ple.ple_embedding.n if pl else None
        states = []
        hs = [m.embed(torch.tensor(s, device=dev)[None]) for s in seqs]          # per row [1, T_r, hc d]
        pes = {}
        for i, L in enumerate(m.layers):
            _on(L, i, a, dev)
            st = {}
            if L.kind != "linear_attention":
                st["kv"] = KV(B, self.kvh, Lmax, self.hd, dev, hs[0].dtype)
            convs, recs, ples = [], [], []
            for r, s in enumerate(seqs):
                T = len(s)
                if T not in pes:
                    pes[T] = m.rope(T, dev)
                sr = {}
                if L.kind == "linear_attention":
                    sr["lin"] = LinCache(L.linear_attn.layer_idx)
                else:
                    st["kv"].row = r; sr["kv"] = st["kv"]
                if L.ple is not None:
                    sr["ple"] = torch.zeros(1, L.ple.conv1d.in_channels, L.ple.pad, device=dev, dtype=hs[r].dtype)
                hs[r] = layer_step(L, i, hs[r], pes[T], torch.tensor(s, device=dev)[None], sr, None)
                if L.kind == "linear_attention":
                    sl = sr["lin"].layers[L.linear_attn.layer_idx]
                    convs.append(sl.conv_states[0]); recs.append(sl.recurrent_states[0])
                if L.ple is not None:
                    ples.append(sr["ple"])
            if L.kind == "linear_attention":
                st["lin"] = LinCache(L.linear_attn.layer_idx, torch.cat(convs), torch.cat(recs))
            else:
                st["kv"].row = None
            if L.ple is not None:
                st["ple"] = torch.cat(ples)
            states.append(st)
            _off(L, i, a)
        W = m.lm_head.weight
        last = torch.cat([m.head_in(h[:, -1:]) for h in hs])                    # [B, 1, d]
        nxt = (last[:, 0].float() @ W.float().T).argmax(-1)
        gen = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=dev)
        ids_hist = [list(s) for s in seqs]
        pos = lens.clone()                                                       # the next token's position
        ar = torch.arange(Lmax, device=dev)
        t0 = time.time()
        for step in range(max_new):
            for r in range(B):
                if not done[r]:
                    tok = int(nxt[r])
                    if tok in self.eos:
                        done[r] = True
                    else:
                        gen[r].append(tok); ids_hist[r].append(tok)
            if check and step >= check:
                break
            if bool(done.all()) or step == max_new - 1:
                break
            if step % 32 == 0:
                print(f"    step {step}: {int((~done).sum())} live, {time.time() - t0:.0f} s, max mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            x_ids = torch.tensor([ids_hist[r][-1] if not done[r] else self._eos0 for r in range(B)], device=dev)[:, None]
            h = m.embed(x_ids)
            pe = m.rotary(h, pos[:, None])
            lim = int(pos.max()) + 1
            mask = (ar[None, :lim] <= pos[:, None])[:, None, None, :]           # [B, 1, 1, lim]
            ple_ids = torch.tensor([(ids_hist[r] if not done[r] else ids_hist[r] + [self._eos0])[-n:] for r in range(B)],
                                   device=dev) if n else None
            for i, L in enumerate(m.layers):
                _on(L, i, a, dev)
                st = states[i]
                if "kv" in st:
                    st["kv"].pos, st["kv"].lim = pos, lim
                h = layer_step(L, i, h, pe, ple_ids, st, mask)
                _off(L, i, a)
            nxt = (m.head_in(h)[:, 0].float() @ W.float().T).argmax(-1)
            if check:
                self._check(seqs, ids_hist, done, h, step)
            pos = pos + 1          # (a finished row feeds EOS from here on: harmless, ignored)
        return gen

    @property
    def _eos0(self):
        return next(iter(self.eos))

    def _full(self, full, pad):
        """last-position logits of each id list by the full recompute (q38_eval_lab's path), rows batched"""
        m, a, dev = self.m, self.a, self.dev
        W = m.lm_head.weight
        T = max(len(s) for s in full); T = (T + pad - 1) // pad * pad
        ids = torch.zeros(len(full), T, dtype=torch.long)
        for j, s in enumerate(full):
            ids[j, :len(s)] = torch.tensor(s)
        pe = m.rope(T, dev)
        gi = ids.to(dev); hh = m.embed(gi)
        for i, L in enumerate(m.layers):
            _on(L, i, a, dev)
            hh = L(hh, pe, gi)
            _off(L, i, a)
        x = m.head_in(hh)
        return torch.stack([x[j, len(full[j]) - 1] for j in range(len(full))]).float() @ W.float().T

    def _check(self, seqs, ids_hist, done, h, step):
        """the cached step's logits against the full recompute on the same ids; noise floor: the full recompute with
        the rows batched vs one row at a time (other GEMM shapes, so other roundings and near-tie routings)"""
        m = self.m
        W = m.lm_head.weight
        lc = m.head_in(h)[:, 0].float() @ W.float().T
        rows = [r for r in range(len(seqs)) if not done[r]]
        full = [ids_hist[r] for r in rows]
        lb = self._full(full, 64)
        ls = torch.cat([self._full([f], 128) for f in full])
        lc = lc[rows]
        f = lambda u, v: ((u - v).abs().max().item(), (u.argmax(-1) == v.argmax(-1)).float().mean().item())
        print(f"  check step {step}: cached vs full {f(lc, lb)}; full batched vs single (floor) {f(lb, ls)}; "
              f"cached vs single {f(lc, ls)}; max |logit| {lb.abs().max().item():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slim", default=None)
    ap.add_argument("--fp", default=None, help="the fp teacher: the release's expert shards (q38_hf_fetch_exp.py)")
    ap.add_argument("--fp-stack", default=None); ap.add_argument("--fp-pin", type=int, default=0)
    ap.add_argument("--bench", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gsm8k:200,humaneval:164")
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--resident", type=int, default=0); ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(a.model)
    gc = GenerationConfig.from_pretrained(a.model)
    eos = gc.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos])
    if a.fp:
        m = build_fp(a, dev)
    else:
        m = load_slim(a.model, a.slim, dev, layers=a.layers, resident=a.resident)
    _ST[0] = Streamer(m, a, dev)
    if a.fp:
        _FP[0] = FPExperts(m, a, dev)
    g = Gen(m, a, dev, eos)
    os.makedirs(a.out, exist_ok=True)
    for spec in a.tasks.split(","):
        task, lim = spec.split(":"); fn = os.path.join(a.out, f"{task}.jsonl")
        done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) and not a.check else set()
        todo = [(i, p) for i, p in items(a.bench, task, int(lim)) if i not in done]
        enc = {i: tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)).input_ids for i, p in todo}
        todo.sort(key=lambda x: len(enc[x[0]]))
        print(f"{task}: {len(todo)} to do ({len(done)} done), batch {a.bs}, max_new {a.max_new}", flush=True)
        t0 = time.time()
        for b in range(0, len(todo), a.bs):
            chunk = todo[b:b + a.bs]; t = time.time()
            gen = g.run([enc[i] for i, _ in chunk], a.max_new, check=a.check)
            if a.check:
                return
            with open(fn, "a") as fo:
                for (i, p), gi in zip(chunk, gen):
                    fo.write(json.dumps({"id": i, "prompt": p, "output": tok.decode(gi, skip_special_tokens=True),
                                         "gen_tokens": len(gi)}) + "\n")
            print(f"  {task} batch {b // a.bs + 1}/{(len(todo) + a.bs - 1) // a.bs}: {len(chunk)} items, "
                  f"{max(len(x) for x in gen)} max tokens, {time.time() - t:.0f} s", flush=True)
        print(f"{task} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
