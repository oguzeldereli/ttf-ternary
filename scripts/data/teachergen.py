"""Teacher-generated text with varied openings for the broad-mix test (wiki + real chat / instruct + code + teacher-generated
text, varied openings). The full-precision Qwen3.5-0.8B-Base continues openings drawn from the
training streams (an opening = the first 8-64 tokens of a random window of one of them, cut at a random length), or
starts from <|endoftext|> alone (share --bos), sampling at temperature 1 (the teacher's own distribution) to --seq
tokens per window. Writes uint32 tokens, appending (a restart continues; --n counts windows already written).
  python scripts/data/teachergen.py HF_DIR OUT.bin --n 2000 --open data/wikiqw_train40.bin data/chatqw_train.bin ...
"""
import argparse, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("hf_dir"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--seed", type=int, default=2110)
ap.add_argument("--bos", type=float, default=0.2); ap.add_argument("--open", nargs="+", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.hf_dir)
m = AutoModelForCausalLM.from_pretrained(a.hf_dir, dtype=torch.bfloat16).cuda().eval()
bos = tok.convert_tokens_to_ids("<|endoftext|>")
src = [np.memmap(p, dtype=np.uint32, mode="r") for p in a.open]
done = os.path.getsize(a.out) // (4 * a.seq) if os.path.exists(a.out) else 0
rng = np.random.default_rng(a.seed + done)
torch.manual_seed(a.seed + done)
print(f"{done} windows already in {a.out}", flush=True)
while done < a.n:
    nb = min(a.batch, a.n - done)
    L = int(rng.integers(8, 65)) if rng.random() >= a.bos else 0     # one opening length per batch (no padding)
    rows = []
    for _ in range(nb):
        if L == 0:
            rows.append([bos])
        else:
            s = src[rng.integers(len(src))]; i = int(rng.integers(0, len(s) - L))
            rows.append([bos] + s[i:i + L].astype(np.int64).tolist())
    x = torch.tensor(rows, dtype=torch.long, device="cuda")
    with torch.no_grad():
        y = m.generate(x, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=a.seq + 1 - x.shape[1],
                       min_new_tokens=a.seq + 1 - x.shape[1], pad_token_id=bos, eos_token_id=None)
    with open(a.out, "ab") as f:
        y[:, 1:a.seq + 1].cpu().numpy().astype(np.uint32).tofile(f)
    done += nb
    print(f"{done}/{a.n} windows (opening {L} tokens)", flush=True)
print(f"{a.out}: {done * a.seq} tokens")
