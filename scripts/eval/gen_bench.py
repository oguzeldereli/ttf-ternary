"""Generation benchmarks for Qwen3.5 / 3.8 models, full precision or TTF-packed, with identical prompts and decoding.

  python scripts/eval/gen_bench.py HF_DIR --fp  --task gsm8k --limit 200 --out OUT.jsonl     # bf16, CPU offload
  python scripts/eval/gen_bench.py HF_DIR --ttf WEIGHTS.pt --task gsm8k --limit 200 --out OUT.jsonl

Tasks (data/bench/*.jsonl, the first --limit items): gsm8k (test split), humaneval (164), ifeval (541). Chat template,
thinking off, greedy decoding, batched with left padding (prompts sorted by length). Writes one line per item (prompt id,
output text, generated tokens) and skips items already in OUT, so a stopped run resumes. Scoring: score_bench.py.
--fp keeps what does not fit --gpu_gib on the GPU in CPU memory (accelerate); decoding then streams those layers
through the GPU once per step for the whole batch, so a large batch costs little more than one sequence.
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

ap = argparse.ArgumentParser()
ap.add_argument("hf_dir"); ap.add_argument("--task", required=True, choices=["gsm8k", "humaneval", "ifeval"])
ap.add_argument("--fp", action="store_true"); ap.add_argument("--ttf", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_new", type=int, default=0, help="default: gsm8k 512, humaneval 512, ifeval 1024")
ap.add_argument("--gpu_gib", type=int, default=14); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert a.fp != bool(a.ttf), "--fp or --ttf WEIGHTS"
max_new = a.max_new or {"gsm8k": 512, "humaneval": 512, "ifeval": 1024}[a.task]

GSM = ("{q}\n\nSolve the problem step by step. End with the final answer on its own line in the form "
       "\"Answer: <number>\".")
HE = ("Complete the following Python function. Reply with the complete function (including the signature and any imports "
      "it needs) in one ```python code block.\n\n```python\n{p}```")


def items():
    f = {"gsm8k": "gsm8k_test.jsonl", "humaneval": "humaneval.jsonl", "ifeval": "ifeval.jsonl"}[a.task]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "bench", f))]
    rows = rows[:a.limit] if a.limit else rows
    for r in rows:
        if a.task == "gsm8k":
            yield r["id"], GSM.format(q=r["question"])
        elif a.task == "humaneval":
            yield r["task_id"], HE.format(p=r["prompt"])
        else:
            yield str(r["key"]), r["prompt"]


from transformers import AutoTokenizer                                    # noqa: E402
tok = AutoTokenizer.from_pretrained(a.hf_dir)
tok.padding_side = "left"
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
todo = [(i, p) for i, p in items() if i not in done]
texts = {i: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False) for i, p in todo}
todo.sort(key=lambda x: len(tok(texts[x[0]]).input_ids))
print(f"{a.task}: {len(todo)} to do ({len(done)} done), max_new {max_new}, batch {a.bs}", flush=True)
if not todo:
    sys.exit(0)

t0 = time.time()
if a.fp:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    hf = Qwen3_5ForCausalLM.from_pretrained(a.hf_dir, dtype=torch.bfloat16, device_map="auto",
                                            max_memory={0: f"{a.gpu_gib}GiB", "cpu": "110GiB"}).eval()
else:
    sys.path.insert(0, os.path.join(ROOT, "scripts", "infer"))
    from ttf_qwen import build
    hf, _ = build(a.hf_dir, a.ttf, "cuda", True)
print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
from transformers import GenerationConfig                                 # noqa: E402
gc = GenerationConfig.from_pretrained(a.hf_dir)   # the model's own stop tokens (<|im_end|> too): ttf_qwen.build() makes
eos = gc.eos_token_id                             # the model from config.json, whose eos is <|endoftext|> only
pad = gc.pad_token_id if gc.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)
print(f"eos {eos}, pad {pad}", flush=True)

with open(a.out, "a") as fo:
    for b in range(0, len(todo), a.bs):
        chunk = todo[b:b + a.bs]
        enc = tok([texts[i] for i, _ in chunk], return_tensors="pt", padding=True).to("cuda")
        t = time.time()
        with torch.no_grad():
            out = hf.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        gen = out[:, enc.input_ids.shape[1]:]
        for (i, p), g in zip(chunk, gen):
            n = int((g != pad).sum())
            fo.write(json.dumps({"id": i, "prompt": p, "output": tok.decode(g, skip_special_tokens=True),
                                 "gen_tokens": n}) + "\n")
        fo.flush()
        print(f"  batch {b // a.bs + 1}/{(len(todo) + a.bs - 1) // a.bs}: {len(chunk)} items, {gen.shape[1]} steps, "
              f"{time.time() - t:.0f} s", flush=True)
print(f"done in {time.time() - t0:.0f} s", flush=True)
