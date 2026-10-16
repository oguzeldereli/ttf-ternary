"""Qwen3.8 token stream -> Qwen3.5 tokens (the tokenizers agree on plain text, Wikipedia bit for bit, but not on all
special tokens): decode pieces cut after each <|im_end|> / <|endoftext|> with the Qwen3.8 tokenizer, re-encode with
Qwen3.5's (special tokens kept as text where the target lacks them).
  python scripts/quant/retok_q38_q35.py data/chatq8_val.bin data/chatqw_val.bin --src_tok ~/qw38tok/tokenizer.json \
      --dst_tok HF_DIR/tokenizer.json [--max_tokens 4e6]
"""
import argparse
import numpy as np
from tokenizers import Tokenizer

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--src_tok", required=True); ap.add_argument("--dst_tok", required=True)
ap.add_argument("--max_tokens", type=float, default=0)
a = ap.parse_args()
S, D = Tokenizer.from_file(a.src_tok), Tokenizer.from_file(a.dst_tok)
d = np.fromfile(a.src, dtype=np.uint32)
if a.max_tokens:
    d = d[:int(a.max_tokens)]
cuts = {S.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>") if S.token_to_id(t) is not None}
ends = np.flatnonzero(np.isin(d, list(cuts))) + 1
out, s0 = [], 0
for e in list(ends) + [len(d)]:
    if e > s0:
        out.append(np.array(D.encode(S.decode(d[s0:e].tolist(), skip_special_tokens=False), add_special_tokens=False).ids, dtype=np.uint32))
    s0 = e
o = np.concatenate(out); o.tofile(a.out)
print(f"{a.src} -> {a.out}: {len(d)} -> {len(o)} tokens")
