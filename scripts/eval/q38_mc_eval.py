"""q38_mc_eval.py: zero-shot multiple-choice benchmarks scored by log-likelihood, the same items, prompts and scoring
for the Q38 student (a ttf.q38.train checkpoint) and the fp teacher (HF), so the two numbers are comparable.

Scoring (as lm-eval-harness): each choice is a continuation of the item's context; acc = argmax of the summed log-prob
of the continuation tokens, acc_norm = argmax of that sum / the continuation's character length.
Tasks (HF datasets, --limit items each, fixed order): arc_easy, arc_challenge, hellaswag, winogrande, boolq, mmlu.

Student: run exactly as python -m ttf.q38.train with the run's flags (+ --resume --out RUN for its ckpt), plus
  --mc-tasks arc_easy,arc_challenge,hellaswag,winogrande,boolq,mmlu --mc-limit 500 --mc-out mc_student.json
  [--mc-seq 512] [--mc-batch 16]
  The forward is the trainer's own layer-streamed path (forward_stash, no stash), then the final norm / lm_head on the
  continuation positions only.
Teacher:
  python scripts/eval/q38_mc_eval.py --hf HF_DIR --mc-tasks ... --mc-limit 500 --mc-out mc_teacher.json
Dry run (tiny CPU model, random items; checks the scoring path against a direct full forward):
  python scripts/eval/q38_mc_eval.py --selftest"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _pop(flag, default, cast=str):
    if flag in sys.argv:
        i = sys.argv.index(flag); v = sys.argv[i + 1]; del sys.argv[i:i + 2]
        return cast(v)
    return default


def _popflag(flag):
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


TASKS = _pop("--mc-tasks", "arc_easy,arc_challenge,hellaswag,winogrande,boolq,mmlu").split(",")
LIMIT = _pop("--mc-limit", 500, int)
OUT = _pop("--mc-out", "mc_eval.json")
SEQ = _pop("--mc-seq", 512, int)
BATCH = _pop("--mc-batch", 16, int)
PADM = _pop("--mc-pad-multiple", 64, int)
HF = _pop("--hf", "")
SELFTEST = _popflag("--selftest")


# ---- items: (task, [(context, continuation), ...], gold index)
def load_items(task, limit):
    from datasets import load_dataset
    out = []
    if task in ("arc_easy", "arc_challenge"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy" if task == "arc_easy" else "ARC-Challenge", split="test")
        for r in ds:
            ctx = f"Question: {r['question']}\nAnswer:"
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels:
                continue
            out.append((task, [(ctx, " " + t) for t in r["choices"]["text"]], labels.index(r["answerKey"])))
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for r in ds:
            ctx = r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()
            out.append((task, [(ctx, " " + e) for e in r["endings"]], int(r["label"])))
    elif task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        for r in ds:
            s = r["sentence"]; i = s.index("_")
            cont = " " + s[i + 1:].strip()
            out.append((task, [(s[:i] + r["option1"], cont), (s[:i] + r["option2"], cont)], int(r["answer"]) - 1))
    elif task == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        for r in ds:
            ctx = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
            out.append((task, [(ctx, " no"), (ctx, " yes")], int(bool(r["answer"]))))
    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0)
        for r in ds:
            opts = "".join(f"\n{L}. {c}" for L, c in zip("ABCD", r["choices"]))
            ctx = f"{r['question'].strip()}{opts}\nAnswer:"
            out.append((task, [(ctx, " " + L) for L in "ABCD"], int(r["answer"])))
    else:
        raise ValueError(task)
    return out[:limit]


def encode(tok, ctx, cont, seq):
    """token ids of ctx + cont (cont tokenised in context, as lm-eval), truncated from the left to seq; returns (ids,
    number of continuation tokens)"""
    a = tok(ctx, add_special_tokens=False)["input_ids"]
    b = tok(ctx + cont, add_special_tokens=False)["input_ids"]
    n = len(b) - len(a)
    if n <= 0 or b[:len(a)] != a:              # (tokenisation merged across the boundary: tokenise cont on its own)
        b = a + tok(cont, add_special_tokens=False)["input_ids"]; n = len(b) - len(a)
    return b[-seq:], min(n, seq - 1)


def score(requests, logprob_fn, batch, pad_id):
    """requests [(ids, ncont)] -> summed log-prob of the last ncont tokens of each; logprob_fn(ids [n, T]) -> [n, T-1, V]
    log-probs (or a callable taking (ids, positions) -> per-position target log-probs, see below)"""
    order = sorted(range(len(requests)), key=lambda i: -len(requests[i][0]))
    res = [None] * len(requests)
    for j in range(0, len(order), batch):
        idx = order[j:j + batch]
        T = max(len(requests[i][0]) for i in idx)
        T = -(-T // PADM) * PADM          # (one length class: the chunked linear-attention path differs below its chunk)
        ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        for r, i in enumerate(idx):                    # right padding: causal, so the real tokens never see it
            s = requests[i][0]; ids[r, :len(s)] = torch.tensor(s)
        spans = [(len(requests[i][0]) - requests[i][1], len(requests[i][0])) for i in idx]
        lp = logprob_fn(ids, spans)                    # list of [ncont] target log-probs
        for r, i in enumerate(idx):
            res[i] = float(lp[r].sum())
    return res


def run_tasks(tok, logprob_fn, pad_id, tag, items=None):
    out = {"tag": tag, "limit": LIMIT, "seq": SEQ, "time": time.ctime(), "tasks": {}}
    for task in (TASKS if items is None else sorted({it[0] for it in items})):
        t0 = time.time()
        its = load_items(task, LIMIT) if items is None else [it for it in items if it[0] == task]
        reqs, owner = [], []
        for k, (_, pairs, gold) in enumerate(its):
            for c, (ctx, cont) in enumerate(pairs):
                reqs.append(encode(tok, ctx, cont, SEQ) if tok is not None else (ctx + cont, len(cont)))
                owner.append((k, c, len(cont) if isinstance(cont, str) else len(cont)))
        lps = score(reqs, logprob_fn, BATCH, pad_id)
        per = {}
        for (k, c, nch), lp in zip(owner, lps):
            per.setdefault(k, []).append((lp, lp / max(1, nch)))
        acc = sum(max(range(len(v)), key=lambda c: v[c][0]) == its[k][2] for k, v in per.items()) / len(per)
        accn = sum(max(range(len(v)), key=lambda c: v[c][1]) == its[k][2] for k, v in per.items()) / len(per)
        out["tasks"][task] = {"n": len(per), "acc": round(acc, 4), "acc_norm": round(accn, 4),
                              "s": round(time.time() - t0, 1), "lp": lps}
        print(f"[{tag}] {task}: n {len(per)} acc {acc:.4f} acc_norm {accn:.4f} ({time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(OUT, "w"))
    return out


# ---- the student: the trainer's layer-streamed forward
def student_logprob_fn(tr):
    m, dev = tr.m, tr.dev

    @torch.no_grad()
    def fn(ids, spans):
        mbs = [{"ids": ids[r:r + 1]} for r in range(ids.shape[0])]
        hs, _, _ = tr.forward_stash(mbs, stash=False)
        W = m.lm_head.weight
        out = []
        for h, (a, b), r in zip(hs, spans, range(len(hs))):
            x = m.head_in(h)[0]                                  # [T, d]
            lg = (x[a - 1:b - 1].float() @ W.float().T)          # positions predicting tokens a .. b-1
            lp = torch.log_softmax(lg, -1)
            out.append(lp.gather(1, ids[r, a:b].to(lp.device)[:, None])[:, 0].cpu())
        return out
    return fn


def student_run(tr, args, batches, val_batches):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    run_tasks(tok, student_logprob_fn(tr), pad, f"student step {tr.step}")


# ---- the teacher: HF
def teacher_main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(HF, torch_dtype="auto", device_map="auto")
    model.eval()
    dev = next(model.parameters()).device

    @torch.no_grad()
    def fn(ids, spans):
        lg = model(input_ids=ids.to(dev)).logits
        out = []
        for r, (a, b) in enumerate(spans):
            lp = torch.log_softmax(lg[r, a - 1:b - 1].float(), -1)
            out.append(lp.gather(1, ids[r, a:b].to(lp.device)[:, None])[:, 0].cpu())
        return out
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    run_tasks(tok, fn, pad, "teacher " + HF)


# ---- self test: the tiny CPU model; the trainer path against a direct full forward of the same student
# the tiny random Q38 student / teacher / KD batches of the former pipeline check (tests/checks history), for selftest
T, MB, K_, S_, STEPS = 48, 4, 16, 16, 3


def _tiny_setup():
    import copy, types
    from ttf.q38 import model as M
    from ttf.q38 import train as QT
    from ttf.data import _kd_sampled
    torch.set_num_threads(2)
    t = dict(hidden_size=64, hc_count=4, hc_lowrank=8, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
             head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
             linear_value_head_dim=16, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
             shared_expert_intermediate_size=16, vocab_size=1000, ngram_vocab_size_base=101, heads_per_ngram=2,
             ple_embed_dim=64, eos_token_id=999, bos_token_id=999, indexer_n_heads=2, indexer_kv_heads=1,
             indexer_head_dim=16, indexer_budget=64, layer_types=["linear_attention"] * 3 + ["full_attention"],
             ple_layer_ids=[2], max_position_embeddings=4096, ngram_size=3, ple_conv_kernel_size=3, hidden_act="silu",
             rms_norm_eps=1e-6)
    cfg = M.text_config({"text_config": t})
    torch.manual_seed(0)
    teacher = M.build(cfg, device="cpu")
    with torch.no_grad():
        for n, p in teacher.named_parameters():
            if n.endswith("A_log"):
                p.copy_(torch.randn_like(p) * 0.1)
            elif p.dim() >= 2:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1)
        for L in teacher.layers:
            if L.ple is not None:
                ng = L.ple.ple_embedding
                ng.table = torch.randn(ng.rows, ng.dim)
        teacher.lm_head.weight.mul_(12.0)

    def batch(seed):
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(MB):
            w = torch.randint(0, 999, (T + 1,), generator=g)
            with torch.no_grad():
                lp = torch.log_softmax(teacher(w[None, :-1]).float()[0], -1)
            top = lp.topk(K_, -1)
            rest = lp.exp().scatter(1, top.indices, 0.0)
            sids = torch.multinomial(rest / rest.sum(-1, keepdim=True), S_, replacement=True, generator=g)
            ids, lw = _kd_sampled(top.indices.numpy(), top.values.numpy(), sids.numpy())
            out.append({"ids": w[None, :-1], "tgt": w[1:], "kd_ids": torch.from_numpy(ids), "kd_lw": torch.from_numpy(lw)})
        return out

    student = copy.deepcopy(teacher)
    for L in student.layers:
        QT.swap_ternary(L, adapter_rank=4, quick=True)
        for n, tm in QT.tern_modules(L):
            tm.ta_packed.copy_(QT.pack3(torch.randint(-1, 2, (tm.E, tm.N, tm.ra), dtype=torch.int8)))
        for n, tm in QT.tern_modules(L):
            tm.float()
    args = types.SimpleNamespace(steps=40, theta=2.0, tau=200.0, sketch=16, rank_ratio=0.5, lr=3e-3, lr_min=3e-4,
                                 warmup=2, kd_weight=1.0, head_chunk=16, expert_chunk=4, pin=False, save_every=10 ** 9,
                                 eval_every=10 ** 9, micro=MB, seq=T, val_micro=MB, pipe_group=3, wbatch=1,
                                 ttf_store="fp32")
    return student, batch, args


def selftest():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ttf.q38 import train as QT
    student, batch, args = _tiny_setup()
    tr = QT.Trainer(student, args, "cpu")
    fn = student_logprob_fn(tr)
    g = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def direct(ids, spans):
        out = []
        for r, (a, b) in enumerate(spans):
            lg = student(ids[r:r + 1]).float()[0]           # (the same padded length as the trainer path)
            lp = torch.log_softmax(lg[a - 1:b - 1], -1)
            out.append(lp.gather(1, ids[r, a:b][:, None])[:, 0])
        return out
    reqs = [(torch.randint(0, 998, (int(torch.randint(8, 40, (1,), generator=g)),), generator=g).tolist(),
             int(torch.randint(1, 6, (1,), generator=g))) for _ in range(23)]
    a = score(reqs, fn, 5, 999)
    b = score(reqs, direct, 1, 999)
    c = score(reqs, fn, 1, 999)
    err_ab = max(abs(x - y) for x, y in zip(a, b)); err_ac = max(abs(x - y) for x, y in zip(a, c))
    print(f"selftest: trainer path (batch 5, padded) vs direct forward max |d logprob| {err_ab:.2e}; "
          f"batch 5 vs batch 1 {err_ac:.2e}", flush=True)
    assert err_ac < 1e-4 and err_ab < 5e-2, "scoring mismatch"   # (err_ab: the trainer path runs rope in bf16, as training does)
    # end to end on fake items: ids as 'strings' through an identity tokenizer
    items = []
    for k in range(12):
        ctx = torch.randint(0, 998, (20,), generator=g).tolist()
        items.append(("fake", [(ctx, torch.randint(0, 998, (3,), generator=g).tolist()) for _ in range(4)], k % 4))

    class Tok:
        def __call__(self, x, add_special_tokens=False):
            return {"input_ids": list(x)}
    global LIMIT
    LIMIT = 100
    o = run_tasks(Tok(), fn, 999, "selftest", items=[(t, [(c, d) for c, d in p], gd) for t, p, gd in items])
    print("selftest ok", o["tasks"]["fake"]["acc"], flush=True)


if __name__ == "__main__":
    if SELFTEST:
        selftest()
    elif HF:
        teacher_main()
    else:
        from ttf.q38 import train as QT
        QT.run = student_run
        QT.main()
