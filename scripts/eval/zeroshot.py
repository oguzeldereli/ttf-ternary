"""Zero-shot evaluation (GPU; run on a lab PC / 4090 / Myriad, never the laptop GPU):

    python scripts/eval/zeroshot.py RUN_DIR [RUN_DIR ...] [--tasks lambada,hellaswag,piqa,arc_easy,arc_challenge,winogrande]

Tasks pre-tokenised by scripts/eval/zeroshot_tok.py (data/zeroshot/tok/TASK.json). Multiple choice: the summed
log-probability of each continuation given its context; acc = argmax, acc_norm = argmax of log-prob per byte of the
continuation (as lm-evaluation-harness). LAMBADA: acc = every target token is the greedy prediction, plus the target's
perplexity. WinoGrande: partial scoring (the option is in the context, the shared continuation is scored). Writes
RUN_DIR/zeroshot.json (accuracies with standard errors) and RUN_DIR/zeroshot.npz (per-example correctness, for paired
comparisons between models).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.bitlinear import STATE                     # noqa: E402
from final_eval import load_model                      # noqa: E402


@torch.no_grad()
def score(model, seqs, device, bs=16, max_len=2048):
    """seqs: list of (ctx ids, cont ids) -> list of (sum log p(cont), all greedy)"""
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        full = [(seqs[i][0] + seqs[i][1])[-(max_len + 1):] for i in idx]
        ncont = [len(seqs[i][1]) for i in idx]
        T = max(len(f) for f in full) - 1
        x = torch.zeros(len(idx), T, dtype=torch.long)
        for j, f in enumerate(full):
            x[j, :len(f) - 1] = torch.tensor(f[:-1])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x.to(device))
        lp = F.log_softmax(logits.float(), -1)
        for j, f in enumerate(full):
            n = len(f) - 1; k = ncont[j]
            tgt = torch.tensor(f[1:], device=device)[n - k:n]
            row = lp[j, n - k:n]
            out[idx[j]] = (row.gather(1, tgt[:, None]).sum().item(), bool((row.argmax(1) == tgt).all().item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tasks", default="lambada,hellaswag,piqa,arc_easy,arc_challenge,winogrande")
    ap.add_argument("--dir", default="data/zeroshot/tok")
    ap.add_argument("--mag_kind", default="add")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU only (never the laptop GPU)"
    STATE.updates_enabled = False
    tasks = {t: json.load(open(os.path.join(a.dir, t + ".json"))) for t in a.tasks.split(",")}
    for run in a.runs:
        t0 = time.time()
        path = next(p for p in (os.path.join(run, f) for f in ("eval.pt", "ckpt.pt", "ckpt_9154.pt")) if os.path.exists(p))
        model, _ = load_model(path, "cuda", a.mag_kind)
        res, arrays = {"run": os.path.basename(os.path.normpath(run))}, {}
        for t, ex in tasks.items():
            if t == "lambada":
                r = score(model, [(e["ctx"], e["cont"][0]) for e in ex], "cuda")
                acc = np.array([g for _, g in r], dtype=float)
                ntok = sum(len(e["cont"][0]) for e in ex)
                res[t] = {"acc": acc.mean(), "acc_se": acc.std(ddof=1) / math.sqrt(len(acc)),
                          "ppl": math.exp(-sum(l for l, _ in r) / ntok), "n": len(acc)}
                arrays[t] = acc
                continue
            seqs, owner = [], []
            for i, e in enumerate(ex):
                for c, k in zip(e["ctxs"], e["conts"]):
                    seqs.append((c, k)); owner.append(i)
            r = score(model, seqs, "cuda")
            lls = [[] for _ in ex]
            for (l, _), i in zip(r, owner):
                lls[i].append(l)
            acc = np.array([float(np.argmax(l) == e["label"]) for l, e in zip(lls, ex)])
            d = {"acc": acc.mean(), "acc_se": acc.std(ddof=1) / math.sqrt(len(acc)), "n": len(acc),
                 "chance": float(np.mean([1 / len(e["conts"]) for e in ex]))}
            if "nbytes" in ex[0]:
                accn = np.array([float(np.argmax(np.array(l) / np.array(e["nbytes"])) == e["label"]) for l, e in zip(lls, ex)])
                d.update(acc_norm=accn.mean(), acc_norm_se=accn.std(ddof=1) / math.sqrt(len(accn)))
                arrays[t + "_norm"] = accn
            res[t] = d; arrays[t] = acc
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(os.path.join(run, "zeroshot.json"), "w"), indent=1)
        np.savez(os.path.join(run, "zeroshot.npz"), **arrays)
        print(res["run"], " ".join(f"{t}: {v['acc']:.3f}" + (f"/{v['acc_norm']:.3f}" if "acc_norm" in v else "")
                                   + (f" ppl {v['ppl']:.1f}" if "ppl" in v else "") for t, v in res.items() if isinstance(v, dict)),
              f"({res['seconds']} s)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
