"""Re-tokenize one of our token streams with another tokenizer, article by article (same text, same train / val split):
decode with the source tokenizer, encode with the target's tokenizer.json, write uint32 with the target's end-of-text
between articles.
    python scripts/data/retok.py data/wiki32k_val.bin data/wikiqw_val.bin --target HF_DIR [--src llama] [--max_tokens N]
"""
import argparse, json, os, time
import numpy as np
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--target", required=True)
ap.add_argument("--src_tok", default="hf-internal-testing/llama-tokenizer"); ap.add_argument("--src_eot", type=int, default=2)
ap.add_argument("--max_tokens", type=float, default=0); ap.add_argument("--batch", type=int, default=512)
a = ap.parse_args()
src = np.memmap(a.src, dtype=np.uint16, mode="r")
if a.max_tokens:
    src = src[:int(a.max_tokens)]
dec = Tokenizer.from_pretrained(a.src_tok)
enc = Tokenizer.from_file(os.path.join(a.target, "tokenizer.json"))
cfg = json.load(open(os.path.join(a.target, "config.json")))
eot = cfg.get("eos_token_id", cfg.get("text_config", {}).get("eos_token_id"))
eot = eot[0] if isinstance(eot, list) else eot
cuts = np.flatnonzero(np.asarray(src) == a.src_eot)
starts = np.concatenate([[0], cuts + 1]); ends = np.concatenate([cuts, [len(src)]])
t0 = time.time(); n_in = n_out = 0
with open(a.out, "wb") as f:
    for b in range(0, len(starts), a.batch):
        arts = [np.asarray(src[s:e]).tolist() for s, e in zip(starts[b:b + a.batch], ends[b:b + a.batch]) if e > s]
        texts = dec.decode_batch(arts, skip_special_tokens=True)
        out = []
        for ids in enc.encode_batch(texts, add_special_tokens=False):
            out.extend(ids.ids); out.append(eot)
        np.asarray(out, dtype=np.uint32).tofile(f)
        n_in += sum(len(x) for x in arts); n_out += len(out)
        if (b // a.batch) % 50 == 0:
            print(f"  {b}/{len(starts)} articles, {n_in / 1e6:.1f}M -> {n_out / 1e6:.1f}M tokens, {time.time() - t0:.0f}s", flush=True)
print(f"{a.src} -> {a.out}: {n_in} -> {n_out} tokens (eot {eot}), {time.time() - t0:.0f}s")
