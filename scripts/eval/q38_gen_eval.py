"""q38_gen_eval.py: generation
benchmarks for a Q38 checkpoint with gen_bench.py's prompts and decoding (chat template, thinking off, greedy,
max_new 512), through the trainer's own forward (forward_stash, no KV cache: every new token recomputes the sequence,
so finished sequences leave the batch at once). Output lines as gen_bench.py ({"id", "prompt", "output", "gen_tokens"});
score with ttf-core scripts/eval/score_bench.py.

Run as python -m ttf.q38.train with the run's flags (no --resume), plus
  --gen-slim DIR (a slim checkpoint: L*.pt with blob["layer"], meta slim true) --gen-tasks gsm8k:200,humaneval:164
  --gen-out DIR --gen-bench ROOT/data/bench [--gen-bs 64] [--gen-max-new 512] [--gen-pad 64]
Padding: right padding to a multiple of --gen-pad (the chunked linear attention takes another path below its chunk);
the model is causal, so the logits at each sequence's last real token do not see the padding."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _pop(flag, default, cast=str):
    if flag in sys.argv:
        i = sys.argv.index(flag); v = sys.argv[i + 1]; del sys.argv[i:i + 2]
        return cast(v)
    return default


SLIM = _pop("--gen-slim", "")
TASKS = _pop("--gen-tasks", "gsm8k:200,humaneval:164")
OUT = _pop("--gen-out", "gen_out")
BENCH = _pop("--gen-bench", "")
BS = _pop("--gen-bs", 64, int)
MAXNEW = _pop("--gen-max-new", 512, int)
PADM = _pop("--gen-pad", 64, int)

GSM = ("{q}\n\nSolve the problem step by step. End with the final answer on its own line in the form "
       "\"Answer: <number>\".")
HE = ("Complete the following Python function. Reply with the complete function (including the signature and any imports "
      "it needs) in one ```python code block.\n\n```python\n{p}```")


def items(task, limit):
    f = {"gsm8k": "gsm8k_test.jsonl", "humaneval": "humaneval.jsonl"}[task]
    rows = [json.loads(l) for l in open(os.path.join(BENCH, f))][:limit]
    for r in rows:
        yield (r["id"], GSM.format(q=r["question"])) if task == "gsm8k" else (r["task_id"], HE.format(p=r["prompt"]))


def load_slim(tr, d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    for i, L in enumerate(tr.m.layers):
        blob = torch.load(os.path.join(d, f"L{i}.pt"), map_location="cpu", weights_only=False)
        L.load_state_dict(blob["layer"])
    tr.step = meta["step"]
    print(f"slim checkpoint {d}: step {meta['step']}", flush=True)


@torch.no_grad()
def next_tokens(tr, seqs):
    """greedy next token for each id list in seqs (one forward, right padded to a common multiple of PADM)"""
    m = tr.m
    T = max(len(s) for s in seqs); T = (T + PADM - 1) // PADM * PADM
    ids = torch.zeros(len(seqs), T, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
    hs, _, _ = tr.forward_stash([{"ids": ids[r:r + 1]} for r in range(len(seqs))], stash=False)
    W = m.lm_head.weight
    out = []
    for h, s in zip(hs, seqs):
        x = m.head_in(h)[0][len(s) - 1:len(s)]
        out.append(int((x.float() @ W.float().T).argmax(-1)))
    return out


def gen_run(tr, args, batches, val_batches):
    from transformers import AutoTokenizer, GenerationConfig
    if SLIM:
        load_slim(tr, SLIM)
    tok = AutoTokenizer.from_pretrained(args.model)
    gc = GenerationConfig.from_pretrained(args.model)
    eos = gc.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos])
    os.makedirs(OUT, exist_ok=True)
    for spec in TASKS.split(","):
        task, lim = spec.split(":"); fn = os.path.join(OUT, f"{task}.jsonl")
        done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
        todo = [(i, p) for i, p in items(task, int(lim)) if i not in done]
        enc = {i: tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)).input_ids for i, p in todo}
        todo.sort(key=lambda x: len(enc[x[0]]))
        print(f"{task}: {len(todo)} to do ({len(done)} done), batch {BS}, max_new {MAXNEW}", flush=True)
        t0 = time.time()
        with open(fn, "a") as fo:
            for b in range(0, len(todo), BS):
                chunk = todo[b:b + BS]; t = time.time()
                seqs = {i: list(enc[i]) for i, _ in chunk}; gen = {i: [] for i, _ in chunk}; live = [i for i, _ in chunk]
                for _ in range(MAXNEW):
                    if not live:
                        break
                    nt = next_tokens(tr, [seqs[i] for i in live])
                    keep = []
                    for i, n in zip(live, nt):
                        if n in eos:
                            continue
                        seqs[i].append(n); gen[i].append(n); keep.append(i)
                    live = keep
                for i, p in chunk:
                    fo.write(json.dumps({"id": i, "prompt": p, "output": tok.decode(gen[i], skip_special_tokens=True),
                                         "gen_tokens": len(gen[i])}) + "\n")
                fo.flush()
                print(f"  {task} batch {b // BS + 1}/{(len(todo) + BS - 1) // BS}: {len(chunk)} items, "
                      f"{max(len(g) for g in gen.values())} steps, {time.time() - t:.0f} s", flush=True)
        print(f"{task} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    from ttf.q38 import train as QT
    QT.run = gen_run
    QT.main()
