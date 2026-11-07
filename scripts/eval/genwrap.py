"""ttf.train as a generation-benchmark host: the trainer builds and loads the model exactly as for training (--init-ckpt,
ttfconv.fold / adapters, int-k embedding / head), and in place of its eval runs scripts/eval/gen_bench.py's tasks on
model.hf.generate: the same prompts, chat template with thinking off, greedy, the model's own stop tokens, batched with
left padding (prompts sorted by length); one jsonl per task, resumable. The int-k head is dequantized once (bf16 cache).
  GEN_TASKS=gsm8k:200,humaneval:164,ifeval:200 GEN_OUT=DIR GEN_BS=32 HF=... torchrun ... scripts/eval/genwrap.py <trainer args>"""
import json, os, sys, time
import torch
import ttf.intk_emb as IE
import ttf.train as T

_init = IE.IntKMatrix.__init__
IE.IntKMatrix.__init__ = lambda self, W, *a, **k: _init(self, W.cpu() if W.is_cuda else W, *a, **k)   # (CPU temps)

GSM = ("{q}\n\nSolve the problem step by step. End with the final answer on its own line in the form "
       "\"Answer: <number>\".")
HE = ("Complete the following Python function. Reply with the complete function (including the signature and any imports "
      "it needs) in one ```python code block.\n\n```python\n{p}```")
MAXNEW = {"gsm8k": 512, "humaneval": 512, "ifeval": 1024}
FILES = {"gsm8k": "gsm8k_test.jsonl", "humaneval": "humaneval.jsonl", "ifeval": "ifeval.jsonl"}


def items(root, task, limit):
    rows = [json.loads(l) for l in open(os.path.join(root, "data", "bench", FILES[task]))][:limit]
    for r in rows:
        if task == "gsm8k":
            yield r["id"], GSM.format(q=r["question"])
        elif task == "humaneval":
            yield r["task_id"], HE.format(p=r["prompt"])
        else:
            yield str(r["key"]), r["prompt"]


def _gen(model, data, tc, device):
    from transformers import AutoTokenizer, GenerationConfig
    hfd, root, out_dir, bs = os.environ["HF"], os.environ["GEN_ROOT"], os.environ["GEN_OUT"], int(os.environ.get("GEN_BS", 32))
    os.makedirs(out_dir, exist_ok=True)
    hf = model.hf; model.eval(); hf.eval()
    if isinstance(hf.lm_head, IE._Head):      # the int-k head: dequantized once, not per decode step
        with torch.no_grad():
            Wh = hf.lm_head.m.leaf().to(torch.bfloat16).contiguous()
        hf.lm_head.forward = lambda x: torch.nn.functional.linear(x.to(torch.bfloat16), Wh)
    tok = AutoTokenizer.from_pretrained(hfd); tok.padding_side = "left"
    gc = GenerationConfig.from_pretrained(hfd); eos = gc.eos_token_id
    pad = gc.pad_token_id if gc.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)
    print(f"genwrap: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, eos {eos}, pad {pad}", flush=True)
    for spec in os.environ.get("GEN_TASKS", "gsm8k:200,humaneval:164,ifeval:200").split(","):
        task, lim = spec.split(":"); lim = int(lim); fo_p = os.path.join(out_dir, f"{task}.jsonl")
        done = {json.loads(l)["id"] for l in open(fo_p)} if os.path.exists(fo_p) else set()
        todo = [(i, p) for i, p in items(root, task, lim) if i not in done]
        texts = {i: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) for i, p in todo}
        todo.sort(key=lambda x: len(tok(texts[x[0]]).input_ids))
        print(f"genwrap {task}: {len(todo)} to do ({len(done)} done), max_new {MAXNEW[task]}, batch {bs}", flush=True)
        t0 = time.time()
        with open(fo_p, "a") as fo:
            for b in range(0, len(todo), bs):
                chunk = todo[b:b + bs]; t = time.time()
                enc = tok([texts[i] for i, _ in chunk], return_tensors="pt", padding=True).to(device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    o = hf.generate(**enc, max_new_tokens=MAXNEW[task], do_sample=False, eos_token_id=eos, pad_token_id=pad)
                gen = o[:, enc.input_ids.shape[1]:]
                for (i, p), g in zip(chunk, gen):
                    fo.write(json.dumps({"id": i, "prompt": p, "output": tok.decode(g, skip_special_tokens=True),
                                         "gen_tokens": int((g != pad).sum())}) + "\n")
                fo.flush()
                print(f"  {task} batch {b // bs + 1}/{(len(todo) + bs - 1) // bs}: {len(chunk)} items, {gen.shape[1]} steps, "
                      f"{time.time() - t:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        print(f"genwrap {task} done in {time.time() - t0:.0f} s", flush=True)
    return float("nan")


T.evaluate = _gen
T.cli()
