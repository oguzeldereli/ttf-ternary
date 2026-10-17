"""Zero-shot tasks for Qwen3.5 / 3.8 models, full precision or TTF-converted, scored identically (summed log-probability of
each continuation; acc_norm per byte; LAMBADA greedy accuracy; WinoGrande partial scoring, as scripts/eval/zeroshot.py).

  python scripts/eval/qwen_zeroshot.py HF_DIR --fp [--limit 2000]          # full precision, streamed through the GPU
  python scripts/eval/qwen_zeroshot.py HF_DIR --ttf WEIGHTS.pt [--limit 2000]   # a ttf_qwen.py / pack_ttf.py file

--fp keeps the bf16 weights in pinned host memory and runs layer-major (each decoder layer copied to the GPU once per
super-batch of sequences), so a 27B model is scored on a 24 GB card. Sequences are right-padded (every layer is causal, so
the padding never reaches a scored position). Tasks from data/zeroshot/TASK.jsonl, tokenised with the model's tokenizer
(no BOS: Qwen has none; a sequence starts with the context). Writes OUT.json with accuracies and standard errors."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TASKS = ("lambada", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande")


def tokenize(tok, task, limit):
    enc = lambda s: tok(s, add_special_tokens=False).input_ids

    def split(ctx, cont):
        a, b = enc(ctx), enc(ctx + cont)
        k = len(a)
        while k > 0 and b[:k] != a[:k]:
            k -= 1
        return b[:k], b[k:]
    out = []
    for n, l in enumerate(open(f"data/zeroshot/{task}.jsonl")):
        if limit and n >= limit:
            break
        x = json.loads(l)
        if task == "lambada":
            c, t = split(x["context"], x["target"]); out.append({"ctx": c, "cont": [t], "label": 0})
        elif task == "winogrande":
            pairs = [split(cx, x["target"]) for cx in x["contexts"]]
            out.append({"ctxs": [p[0] for p in pairs], "conts": [p[1] for p in pairs], "label": x["label"]})
        else:
            pairs = [split(x["context"], ch) for ch in x["choices"]]
            out.append({"ctxs": [p[0] for p in pairs], "conts": [p[1] for p in pairs], "label": x["label"],
                        "nbytes": [len(ch.encode("utf-8")) for ch in x["choices"]]})
    return out


class StreamedFP:
    """transformers' Qwen3.5 decoder in bf16, weights in pinned host memory, layer-major over a super-batch"""

    def __init__(self, hf_dir, dev="cuda"):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        from ttf.qwen import text_config, _safetensor_index
        from safetensors import safe_open
        self.mq, self.dev = mq, dev
        t = text_config(hf_dir); t._attn_implementation = "sdpa"; self.t = t
        with torch.device("meta"):
            hf = mq.Qwen3_5ForCausalLM(t)
        files = _safetensor_index(hf_dir); handles = {}

        def get(name):
            k, f = files[name]
            if f not in handles:
                handles[f] = safe_open(f, "pt", device="cpu")
            return handles[f].get_tensor(k)
        hf.model.norm.to_empty(device=dev); hf.model.norm.weight.data.copy_(get("model.norm.weight"))
        self.head = get("lm_head.weight" if "lm_head.weight" in files else "model.embed_tokens.weight").to(dev, torch.bfloat16)
        self.emb = get("model.embed_tokens.weight").to(torch.bfloat16)
        hf.model.rotary_emb = type(hf.model.rotary_emb)(t).to(dev)
        self.host = []
        for li, layer in enumerate(hf.model.layers):
            ps = {}
            for n, p in list(layer.named_parameters()):
                ps[n] = get(f"model.layers.{li}.{n}").to(torch.bfloat16).pin_memory()
                mod, _, pn = n.rpartition(".")
                (layer.get_submodule(mod) if mod else layer)._parameters[pn] = torch.nn.Parameter(ps[n], requires_grad=False)
            self.host.append(ps)
        self.hf = hf

    @torch.no_grad()
    def logprobs(self, batches):
        """batches: list of LongTensor [b, T] (right-padded); -> list of log-softmax [b, T, V] fp32 (on the GPU, one at a time
        through a generator, since all of them would not fit)"""
        mq, t, dev = self.mq, self.t, self.dev
        Hs = [self.emb[x].to(dev) for x in batches]
        kws = []
        for x in batches:
            b, T = x.shape
            x0 = torch.zeros(b, T, t.hidden_size, device=dev, dtype=torch.bfloat16)
            pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, b, -1)
            mk = {"config": t, "inputs_embeds": x0, "attention_mask": None, "past_key_values": None, "position_ids": pos[0]}
            masks = {"full_attention": mq.create_causal_mask(**mk), "linear_attention": mq.create_recurrent_attention_mask(**mk)}
            kws.append((self.hf.model.rotary_emb(x0, pos[1:]), masks, pos[0]))
            del x0
        for li, layer in enumerate(self.hf.model.layers):
            for n, p in layer.named_parameters():
                p.data = self.host[li][n].to(dev, non_blocking=True)
            for j in range(len(batches)):
                pe, masks, tp = kws[j]
                o = layer(Hs[j], position_embeddings=pe, attention_mask=masks[t.layer_types[li]], position_ids=tp)
                Hs[j] = o[0] if isinstance(o, tuple) else o
            for n, p in layer.named_parameters():
                p.data = self.host[li][n]
        for j in range(len(batches)):
            yield torch.log_softmax((self.hf.model.norm(Hs[j]) @ self.head.t()).float(), -1)
            Hs[j] = None


class TTFModel:
    def __init__(self, hf_dir, weights, dev="cuda"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "infer"))
        from ttf_qwen import build
        self.hf, _ = build(hf_dir, weights, dev, True)
        self.dev = dev

    @torch.no_grad()
    def logprobs(self, batches):
        for x in batches:
            yield torch.log_softmax(self.hf(input_ids=x.to(self.dev)).logits.float(), -1)


def score(model, seqs, mb=8, super_tokens=262144):
    """seqs: list of (ctx, cont) id lists -> list of (sum log p(cont), all greedy)"""
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    groups, cur, ntok = [], [], 0                  # micro-batches of similar length, grouped into super-batches
    for s in range(0, len(order), mb):
        idx = order[s:s + mb]
        T = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx) - 1
        cur.append(idx); ntok += T * len(idx)
        if ntok >= super_tokens:
            groups.append(cur); cur, ntok = [], 0
    if cur:
        groups.append(cur)
    for g in groups:
        batches = []
        for idx in g:
            full = [seqs[i][0] + seqs[i][1] for i in idx]
            T = max(len(f) for f in full) - 1
            x = torch.zeros(len(idx), T, dtype=torch.long)
            for j, f in enumerate(full):
                x[j, :len(f) - 1] = torch.tensor(f[:-1])
            batches.append(x)
        for idx, lp in zip(g, model.logprobs(batches)):
            for j, i in enumerate(idx):
                f = seqs[i][0] + seqs[i][1]; n = len(f) - 1; k = len(seqs[i][1])
                tgt = torch.tensor(f[1:], device=lp.device)[n - k:n]
                row = lp[j, n - k:n]
                out[i] = (row.gather(1, tgt[:, None]).sum().item(), bool((row.argmax(1) == tgt).all().item()))
            del lp
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir"); ap.add_argument("--fp", action="store_true"); ap.add_argument("--ttf", default="")
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", default=""); ap.add_argument("--mb", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    model = StreamedFP(a.hf_dir) if a.fp else TTFModel(a.hf_dir, a.ttf)
    res = {"model": "fp" if a.fp else a.ttf, "limit": a.limit}
    for task in a.tasks.split(","):
        t0 = time.time()
        ex = tokenize(tok, task, a.limit)
        if task == "lambada":
            r = score(model, [(e["ctx"], e["cont"][0]) for e in ex], a.mb)
            acc = np.array([g for _, g in r], dtype=float)
            ntok = sum(len(e["cont"][0]) for e in ex)
            res[task] = {"acc": acc.mean(), "acc_se": acc.std(ddof=1) / math.sqrt(len(acc)),
                         "ppl": math.exp(-sum(l for l, _ in r) / ntok), "n": len(acc)}
        else:
            seqs, owner = [], []
            for i, e in enumerate(ex):
                for c, k in zip(e["ctxs"], e["conts"]):
                    seqs.append((c, k)); owner.append(i)
            r = score(model, seqs, a.mb)
            lls = [[] for _ in ex]
            for (l, _), i in zip(r, owner):
                lls[i].append(l)
            acc = np.array([float(np.argmax(l) == e["label"]) for l, e in zip(lls, ex)])
            d = {"acc": acc.mean(), "acc_se": acc.std(ddof=1) / math.sqrt(len(acc)), "n": len(acc)}
            if "nbytes" in ex[0]:
                accn = np.array([float(np.argmax(np.array(l) / np.array(e["nbytes"])) == e["label"]) for l, e in zip(lls, ex)])
                d.update(acc_norm=accn.mean(), acc_norm_se=accn.std(ddof=1) / math.sqrt(len(accn)))
            res[task] = d
        v = res[task]
        print(f"{task}: acc {v['acc']:.4f}" + (f" acc_norm {v['acc_norm']:.4f}" if "acc_norm" in v else "")
              + (f" ppl {v['ppl']:.2f}" if "ppl" in v else "") + f" (n {v['n']}, {time.time() - t0:.0f}s)", flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
