"""q38_eval_lab.py: GSM8K / HumanEval generation for a Q38 slim checkpoint on a 24 GB lab GPU, with no Trainer
(no TTF state): the model is built as in ttf.q38.train.main (base float weights from the slim base checkpoint, the layers
swapped to ternary), then each layer's state comes from the slim checkpoint (L*.pt, blob["layer"]). Layers live on
the host; each forward moves layer i to the GPU, runs it, moves it back (--resident N: the first N layers stay).
Prompts and decoding as gen_bench.py / q38_gen_eval.py: chat template, thinking off, greedy, max_new 512; no KV cache
(every new token recomputes the sequence; finished sequences leave the batch at once). Right padding to a multiple of
--pad: the model is causal, so the last real token never sees the padding.

  python scripts/eval/q38_eval_lab.py --model BASE --slim DIR --bench DIR --out DIR [--tasks gsm8k:200,humaneval:164]
         [--bs 64] [--sub 16] [--max-new 512] [--pad 64] [--resident 0] [--compile-hc]
Output: OUT/<task>.jsonl, lines {"id", "prompt", "output", "gen_tokens"} (score with ttf-core score_bench.py)."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38.load import load_slim

GSM = ("{q}\n\nSolve the problem step by step. End with the final answer on its own line in the form "
       "\"Answer: <number>\".")
HE = ("Complete the following Python function. Reply with the complete function (including the signature and any imports "
      "it needs) in one ```python code block.\n\n```python\n{p}```")


def items(bench, task, limit):
    f = {"gsm8k": "gsm8k_test.jsonl", "humaneval": "humaneval.jsonl"}[task]
    rows = [json.loads(l) for l in open(os.path.join(bench, f))][:limit]
    for r in rows:
        yield (r["id"], GSM.format(q=r["question"])) if task == "gsm8k" else (r["task_id"], HE.format(p=r["prompt"]))


@torch.no_grad()
def forward_last(m, a, dev, seqs):
    """hidden at each sequence's last real token after all layers -> greedy next token"""
    T = max(len(s) for s in seqs); T = (T + a.pad - 1) // a.pad * a.pad
    ids = torch.zeros(len(seqs), T, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
    pe = m.rope(T, dev)
    groups = [list(range(j, min(len(seqs), j + a.sub))) for j in range(0, len(seqs), a.sub)]
    gi = [ids[g].to(dev) for g in groups]
    hs = [m.embed(x) for x in gi]
    for i, L in enumerate(m.layers):
        if i >= a.resident:
            L.to(dev)
        for _, t in tern_modules(L):
            t.invalidate()
        hs = [L(h, pe, x) for h, x in zip(hs, gi)]
        for _, t in tern_modules(L):
            t._wc = None
        if i >= a.resident:
            L.to("cpu")
    W = m.lm_head.weight
    out = []
    for g, h in zip(groups, hs):
        x = m.head_in(h)                                   # [b, T, d]
        last = torch.tensor([len(seqs[r]) - 1 for r in g], device=dev)
        xl = x[torch.arange(len(g), device=dev), last]     # [b, d]
        out += (xl.float() @ W.float().T).argmax(-1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slim", required=True)
    ap.add_argument("--bench", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gsm8k:200,humaneval:164")
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512); ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--resident", type=int, default=0); ap.add_argument("--compile-hc", action="store_true")
    ap.add_argument("--layers", type=int, default=0, help="smoke test only: the first N layers")
    a = ap.parse_args()
    dev = "cuda"
    if a.compile_hc:
        QM.compile_hc()
    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(a.model)
    gc = GenerationConfig.from_pretrained(a.model)
    eos = gc.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos])
    m = load_slim(a.model, a.slim, dev, layers=a.layers, resident=a.resident)
    os.makedirs(a.out, exist_ok=True)
    for spec in a.tasks.split(","):
        task, lim = spec.split(":"); fn = os.path.join(a.out, f"{task}.jsonl")
        done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
        todo = [(i, p) for i, p in items(a.bench, task, int(lim)) if i not in done]
        enc = {i: tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)).input_ids for i, p in todo}
        todo.sort(key=lambda x: len(enc[x[0]]))
        print(f"{task}: {len(todo)} to do ({len(done)} done), batch {a.bs}, max_new {a.max_new}", flush=True)
        t0 = time.time()
        with open(fn, "a") as fo:
            for b in range(0, len(todo), a.bs):
                chunk = todo[b:b + a.bs]; t = time.time()
                seqs = {i: list(enc[i]) for i, _ in chunk}; gen = {i: [] for i, _ in chunk}; live = [i for i, _ in chunk]
                for s in range(a.max_new):
                    if not live:
                        break
                    nt = forward_last(m, a, dev, [seqs[i] for i in live])
                    keep = []
                    for i, n in zip(live, nt):
                        if n in eos:
                            continue
                        seqs[i].append(n); gen[i].append(n); keep.append(i)
                    live = keep
                    if s % 32 == 0:
                        print(f"    step {s}: {len(live)} live, {time.time() - t:.0f} s", flush=True)
                for i, p in chunk:
                    fo.write(json.dumps({"id": i, "prompt": p, "output": tok.decode(gen[i], skip_special_tokens=True),
                                         "gen_tokens": len(gen[i])}) + "\n")
                fo.flush()
                print(f"  {task} batch {b // a.bs + 1}/{(len(todo) + a.bs - 1) // a.bs}: {len(chunk)} items, "
                      f"{max(len(g) for g in gen.values())} steps, {time.time() - t:.0f} s", flush=True)
        print(f"{task} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
