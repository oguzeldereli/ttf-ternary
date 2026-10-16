"""Self-generated calibration text for a data-free conversion: sample windows from the full-precision model itself
(temperature 1, from <|endoftext|>), no external data. Writes uint32 tokens.
  python scripts/quant/qwen_selfgen.py HF_DIR data/selfqw.bin --n 64 --seq 2048 --batch 16
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("hf_dir"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=64); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seed", type=int, default=4321)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.hf_dir)
m = AutoModelForCausalLM.from_pretrained(a.hf_dir, dtype=torch.bfloat16).cuda().eval()
bos = tok.convert_tokens_to_ids("<|endoftext|>")
torch.manual_seed(a.seed)
out = []
for b0 in range(0, a.n, a.batch):
    nb = min(a.batch, a.n - b0)
    x = torch.full((nb, 1), bos, dtype=torch.long, device="cuda")
    with torch.no_grad():
        y = m.generate(x, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=a.seq,
                       min_new_tokens=a.seq, pad_token_id=bos, eos_token_id=None)
    out.append(y[:, 1:].cpu().numpy().astype(np.uint32)); print(f"{b0 + nb}/{a.n} windows", flush=True)
np.concatenate(out).reshape(-1).tofile(a.out)
print(f"{a.out}: {sum(o.size for o in out)} tokens")
