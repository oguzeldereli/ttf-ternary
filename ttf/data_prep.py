"""Turn raw text into the uint16 token stream the trainer expects.

Default is byte-level (vocab_size=256, zero dependencies) so it runs out of the
box. For a real 32k BPE vocab, pass --tokenizer <hf-name> (needs `tokenizers`
or `transformers`) and set ModelConfig.vocab_size to match.
"""
from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw utf-8 text file")
    ap.add_argument("--out_prefix", default="data/data")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer name; omit for byte-level")
    ap.add_argument("--val_frac", type=float, default=0.0005)
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        raw = f.read()

    if args.tokenizer is None:
        ids = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        vocab = 256
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        ids = np.array(tok.encode(raw.decode("utf-8", "ignore")), dtype=np.uint16)
        vocab = tok.vocab_size

    n_val = max(1, int(len(ids) * args.val_frac))
    import os
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    ids[:-n_val].tofile(args.out_prefix + "_train.bin")
    ids[-n_val:].tofile(args.out_prefix + "_val.bin")
    print(f"tokens: {len(ids):,} | vocab: {vocab} | "
          f"train {len(ids)-n_val:,} / val {n_val:,}")
    print(f"set ModelConfig.vocab_size = {vocab}")


if __name__ == "__main__":
    main()
