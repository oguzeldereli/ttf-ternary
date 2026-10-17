"""Teacher targets for distilling a converted model: the full-precision model's top-k next-token log-probabilities on
fixed training windows, computed on a GPU too small for the model (layer-major streaming: the weights sit in pinned host
memory and each decoder layer is copied to the GPU once per batch of windows, then run over all of them).

  python scripts/quant/teacher_topk.py HF_DIR DATA.bin OUT_DIR [--windows 24000] [--batch 48] [--k 32] [--seq 2048]

Writes OUT_DIR/kd_00000.npz, kd_00001.npz, ... (each: starts [B] int64, ids [B, seq, k] int32, lp [B, seq, k] fp16:
log-probabilities under the full softmax) and OUT_DIR/done when finished; restarts skip finished shards. Window starts
are drawn uniformly (seed 1234) like the trainer's batches; the trainer's --kd_cache reads them in order."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=24000, help="0 with --sequential: all windows"); ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--k", type=int, default=32); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--mb", type=int, default=4, help="windows per layer call")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sequential", action="store_true", help="every non-overlapping window once, shuffled (teacher_gguf.py)")
    ap.add_argument("--claim_dir", default="", help="a directory on a filesystem all machines share (the UCL home): a "
                    "machine computes shard S only after creating CLAIM_DIR/S (mkdir is atomic), so any number of machines "
                    "fill one cache in shard order, joining or leaving at any time")
    ap.add_argument("--part", default="0/1", help="R/N: only the shards with index %% N == R (N machines, one cache)")
    a = ap.parse_args()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    from ttf.qwen import text_config, _safetensor_index
    from safetensors import safe_open
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    t = text_config(a.hf_dir)
    t._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = mq.Qwen3_5ForCausalLM(t)
    files = _safetensor_index(a.hf_dir)
    handles = {}

    def get(name):
        k, f = files[name]
        if f not in handles:
            handles[f] = safe_open(f, "pt", device="cpu")
        return handles[f].get_tensor(k)

    # small parts on the GPU; each decoder layer's tensors pinned on the host
    hf.model.norm.to_empty(device=dev); hf.model.norm.weight.data.copy_(get("model.norm.weight"))
    head = get("lm_head.weight" if "lm_head.weight" in files else "model.embed_tokens.weight").to(dev, torch.bfloat16)
    emb = get("model.embed_tokens.weight").to(torch.bfloat16)                       # host: a lookup per batch
    hf.model.rotary_emb = type(hf.model.rotary_emb)(t).to(dev)
    host = []
    t0 = time.time()
    for li, layer in enumerate(hf.model.layers):
        ps = {}
        for n, p in list(layer.named_parameters()):  # the meta parameter is replaced by one on the pinned host tensor
            ps[n] = get(f"model.layers.{li}.{n}")
            ps[n] = ps[n].to(torch.bfloat16).pin_memory()   # all bf16, as from_pretrained(dtype=bf16) and the student
            mod, _, pn = n.rpartition(".")
            (layer.get_submodule(mod) if mod else layer)._parameters[pn] = torch.nn.Parameter(ps[n], requires_grad=False)
        assert not list(layer.buffers()), "decoder layer buffers are not handled"
        host.append(ps)
    print(f"weights pinned in {time.time() - t0:.0f}s", flush=True)
    data = np.memmap(a.data, dtype=np.uint32, mode="r")
    rng = np.random.default_rng(a.seed)
    if a.sequential:                                # as teacher_gguf.py: the same starts for the same data and seed
        starts_all = rng.permutation((len(data) - 1) // a.seq).astype(np.int64) * a.seq
        starts_all = starts_all[:a.windows] if a.windows else starts_all
    else:
        starts_all = rng.integers(0, len(data) - a.seq - 1, a.windows)
    r_, n_ = map(int, a.part.split("/"))
    assert a.batch % a.mb == 0
    nsh = len(starts_all) // a.batch                # full shards only
    # per-layer keyword arguments for a micro-batch of mb windows (rotary embeddings, the two mask kinds)
    x0 = torch.zeros(a.mb, a.seq, t.hidden_size, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(a.seq, device=dev).view(1, 1, -1).expand(4, a.mb, -1)
    tpos, ppos = pos[0], pos[1:]
    mk = {"config": t, "inputs_embeds": x0, "attention_mask": None, "past_key_values": None, "position_ids": tpos}
    masks = {"full_attention": mq.create_causal_mask(**mk), "linear_attention": mq.create_recurrent_attention_mask(**mk)}
    pe = hf.model.rotary_emb(x0, ppos)
    del x0
    for sh in range(nsh):
        fn = os.path.join(a.out, f"kd_{sh:05d}.npz")
        if os.path.exists(fn) or sh % n_ != r_:
            continue
        if a.claim_dir:
            try:
                os.makedirs(a.claim_dir, exist_ok=True); os.mkdir(os.path.join(a.claim_dir, f"{sh:05d}"))
            except FileExistsError:
                continue
        t1 = time.time()
        st = starts_all[sh * a.batch:(sh + 1) * a.batch]
        ids = np.stack([data[s:s + a.seq].astype(np.int64) for s in st])
        H = emb[torch.from_numpy(ids)].to(dev)                                       # [B, seq, d] bf16
        for li, layer in enumerate(hf.model.layers):
            for n, p in layer.named_parameters():
                p.data = host[li][n].to(dev, non_blocking=True)
            kw = {"position_embeddings": pe, "attention_mask": masks[t.layer_types[li]], "position_ids": tpos}
            with torch.no_grad():
                for b0 in range(0, len(st), a.mb):
                    out = layer(H[b0:b0 + a.mb], **kw)
                    H[b0:b0 + a.mb] = out[0] if isinstance(out, tuple) else out
            for n, p in layer.named_parameters():
                p.data = host[li][n]
        top_i = np.empty((len(st), a.seq, a.k), dtype=np.int32)
        top_l = np.empty((len(st), a.seq, a.k), dtype=np.float16)
        with torch.no_grad():
            for b in range(len(st)):
                hb = hf.model.norm(H[b])
                for c0 in range(0, a.seq, 512):
                    lg = (hb[c0:c0 + 512] @ head.t()).float()
                    lp = torch.log_softmax(lg, -1)
                    v, i = lp.topk(a.k, -1)
                    top_i[b, c0:c0 + 512] = i.int().cpu().numpy(); top_l[b, c0:c0 + 512] = v.half().cpu().numpy()
        del H
        tmp = fn + ".tmp.npz"
        np.savez(tmp, starts=st, ids=top_i, lp=top_l)
        os.replace(tmp, fn)
        cov = float(np.exp(top_l.astype(np.float32)).sum(-1).mean())
        print(f"shard {sh + 1}/{nsh}: {len(st)} windows in {time.time() - t1:.0f}s, top-{a.k} mass {cov:.3f}", flush=True)
    if all(os.path.exists(os.path.join(a.out, f"kd_{sh:05d}.npz")) for sh in range(nsh)):
        open(os.path.join(a.out, "done"), "w").close()


if __name__ == "__main__":
    main()
