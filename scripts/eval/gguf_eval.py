"""A GGUF model, run as shipped through llama.cpp (scripts/eval/llama_score.cpp), on the same windows and zero-shot items
as the full-precision and TTF numbers:
  python scripts/eval/gguf_eval.py MODEL.gguf --tool ./llama_score --ppl data/wikiq8_val.bin:8,data/mixval12.bin:12
  python scripts/eval/gguf_eval.py MODEL.gguf --tool ./llama_score --runval data/wikiq8_val.bin     # train.py's 64 val windows
  python scripts/eval/gguf_eval.py MODEL.gguf --tool ./llama_score --zeroshot --hf_dir HF_DIR [--limit 2000]
--ppl: mean loss over the first N contiguous 2048-token windows (ttf_qwen.py --ppl; FILE:N:STRIDE: windows starting
every STRIDE tokens, e.g. 2049 for a file of exported 2049-token windows such as ttf-lab wip/export_eval27.py's); --runval: the windows of evaluate()
(fp_runval.py); --zeroshot: qwen_zeroshot.py's tokenisation and scoring (sum log-probability, acc_norm per byte, LAMBADA
greedy). Token ids go to llama.cpp directly (no retokenisation).
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("gguf"); ap.add_argument("--tool", required=True); ap.add_argument("--ngl", type=int, default=99)
ap.add_argument("--ppl", default=""); ap.add_argument("--runval", default=""); ap.add_argument("--zeroshot", action="store_true")
ap.add_argument("--hf_dir", default=""); ap.add_argument("--limit", type=int, default=2000)
ap.add_argument("--tasks", default="lambada,hellaswag,piqa,arc_easy,arc_challenge,winogrande")
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--out", default="")
a = ap.parse_args()

proc = subprocess.Popen([a.tool, a.gguf, str(a.ngl), "4096"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
import atexit
atexit.register(proc.kill)      # a failed run must not leave the model loaded on the GPU


def score(seqs):
    """seqs: list of (ctx, cont) id lists -> list of (sum log p(cont), all greedy), as qwen_zeroshot.score"""
    out = []
    for c, k in seqs:
        ids = np.asarray(list(c) + list(k), dtype=np.int32)
        proc.stdin.write(np.array([len(ids), len(k)], dtype=np.int32).tobytes() + ids.tobytes()); proc.stdin.flush()
        l, g = proc.stdout.readline().split()
        out.append((float(l), g == b"1"))
    return out


def windows(path, starts):
    data = np.memmap(path, dtype=np.uint32, mode="r")
    r = score([(data[s:s + 1].tolist(), data[s + 1:s + a.seq + 1].tolist()) for s in starts])   # 2049 tokens, 2048 scored
    return [-l / a.seq for l, _ in r]


res = {"model": a.gguf}
t0 = time.time()
for spec in filter(None, a.ppl.split(",")):
    parts = spec.split(":"); stride = int(parts[2]) if len(parts) > 2 else a.seq; spec = ":".join(parts[:2])
    path, n = spec.rsplit(":", 1)
    ls = windows(path, [i * stride for i in range(int(n))])
    res[os.path.basename(path)] = {"mean": float(np.mean(ls)), "losses": ls}
    print(f"{path}: mean loss {np.mean(ls):.4f} over {len(ls)} windows ({time.time() - t0:.0f} s): "
          + " ".join(f"{v:.4f}" for v in ls), flush=True)
if a.runval:
    import torch
    data = np.memmap(a.runval, dtype=np.uint32, mode="r")
    gen = torch.Generator().manual_seed(1337 + 12345)        # fp_runval.py / evaluate(): batch 16, 4 iters
    starts = [int(i) for _ in range(4) for i in torch.randint(len(data) - a.seq - 1, (16,), generator=gen)]
    ls = windows(a.runval, starts)
    res["runval"] = {"mean": float(np.mean(ls)), "losses": ls, "starts": starts}
    print(f"runval: mean {np.mean(ls):.4f} over {len(ls)} windows ({time.time() - t0:.0f} s)", flush=True)
if a.zeroshot:
    from transformers import AutoTokenizer
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from qwen_zeroshot import tokenize
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    for task in a.tasks.split(","):
        t1 = time.time(); ex = tokenize(tok, task, a.limit)
        if task == "lambada":
            r = score([(e["ctx"], e["cont"][0]) for e in ex])
            acc = np.array([g for _, g in r], dtype=float); ntok = sum(len(e["cont"][0]) for e in ex)
            d = {"acc": acc.mean(), "acc_se": acc.std(ddof=1) / math.sqrt(len(acc)),
                 "ppl": math.exp(-sum(l for l, _ in r) / ntok), "n": len(acc)}
        else:
            seqs, owner = [], []
            for i, e in enumerate(ex):
                for c, k in zip(e["ctxs"], e["conts"]):
                    seqs.append((c, k)); owner.append(i)
            lls = [[] for _ in ex]
            for (l, _), i in zip(score(seqs), owner):
                lls[i].append(l)
            acc = np.array([float(np.argmax(l) == e["label"]) for l, e in zip(lls, ex)])
            d = {"acc": acc.mean(), "acc_se": acc.std(ddof=1) / math.sqrt(len(acc)), "n": len(acc)}
            if "nbytes" in ex[0]:
                accn = np.array([float(np.argmax(np.array(l) / np.array(e["nbytes"])) == e["label"])
                                 for l, e in zip(lls, ex)])
                d.update(acc_norm=accn.mean(), acc_norm_se=accn.std(ddof=1) / math.sqrt(len(accn)))
        res[task] = d
        print(f"{task}: acc {d['acc']:.4f}" + (f" acc_norm {d['acc_norm']:.4f}" if "acc_norm" in d else "")
              + (f" ppl {d['ppl']:.2f}" if "ppl" in d else "") + f" (n {d['n']}, {time.time() - t1:.0f}s)", flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
proc.stdin.write(np.zeros(2, dtype=np.int32).tobytes()); proc.stdin.close(); proc.wait()
