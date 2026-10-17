"""Word-level tokenizer: a jsonl file with a "body" field -> uint16 token stream.

One unit per token: \w+ runs and individual punctuation marks. Vocab = the V-2
most frequent units + <unk> (id 0) + <eos> (id 1, between documents). Rare units
map to <unk>. Never prints document text; writes train/val .bin + vocab.json.
"""
from __future__ import annotations
import json, re, argparse
from collections import Counter
import numpy as np

PAT = re.compile(r"\w+|[^\w\s]")
UNK, EOS = 0, 1


def iter_bodies(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line).get("body") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl file with a 'body' field")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--out_prefix", default="data/tech")
    ap.add_argument("--val_frac", type=float, default=0.005)
    args = ap.parse_args()

    # pass 1: frequencies
    print("pass 1: counting units ...", flush=True)
    c = Counter()
    for b in iter_bodies(args.input):
        c.update(PAT.findall(b))
    # vocab: specials + top (V-2)
    top = [w for w, _ in c.most_common(args.vocab_size - 2)]
    stoi = {"<unk>": UNK, "<eos>": EOS}
    for w in top:
        stoi[w] = len(stoi)
    cov = sum(c[w] for w in top) / max(1, sum(c.values()))
    print(f"vocab {len(stoi):,} | coverage {cov*100:.2f}% | uniq {len(c):,}", flush=True)

    # pass 2: tokenize -> uint16
    print("pass 2: tokenizing ...", flush=True)
    ids = []
    ndoc = 0
    for b in iter_bodies(args.input):
        for tok in PAT.findall(b):
            ids.append(stoi.get(tok, UNK))
        ids.append(EOS)
        ndoc += 1
    arr = np.array(ids, dtype=np.uint16)
    n_val = max(1, int(len(arr) * args.val_frac))

    import os
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    arr[:-n_val].tofile(args.out_prefix + "_train.bin")
    arr[-n_val:].tofile(args.out_prefix + "_val.bin")
    with open(args.out_prefix + "_vocab.json", "w") as f:
        json.dump(stoi, f)
    unk_rate = float((arr == UNK).mean())
    print(f"docs {ndoc:,} | total tokens {len(arr):,} | "
          f"train {len(arr)-n_val:,} / val {n_val:,}")
    print(f"<unk> rate {unk_rate*100:.2f}% | vocab -> {args.out_prefix}_vocab.json")
    print(f"set ModelConfig.vocab_size = {len(stoi)}")


if __name__ == "__main__":
    main()
