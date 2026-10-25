"""Train/val leakage check for a uint16 token stream split at article boundaries (CPU only).

    python scripts/analysis/val_overlap.py data/wiki32k_train.bin data/wiki32k_val.bin [--eot 2] [--n 32]

Reports: val articles whose exact token sequence also occurs as an article in train; the share of val n-grams
(n tokens, default 32) that occur anywhere in train (rolling hash; collisions make this an upper bound).
"""
import argparse, hashlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("train"); ap.add_argument("val")
ap.add_argument("--eot", type=int, default=2)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--chunk", type=int, default=50_000_000)
a = ap.parse_args()
tr = np.memmap(a.train, dtype=np.uint16, mode="r"); va = np.memmap(a.val, dtype=np.uint16, mode="r")


def articles(d):
    cuts = np.flatnonzero(np.asarray(d) == a.eot)
    s = 0
    for c in cuts:
        yield d[s:c]; s = c + 1


vh = {}
for art in articles(va):
    if len(art) >= 16:
        vh[hashlib.blake2b(np.asarray(art).tobytes(), digest_size=16).digest()] = len(art)
dup = 0; dup_tok = 0
for art in articles(tr):
    if len(art) >= 16:
        h = hashlib.blake2b(np.asarray(art).tobytes(), digest_size=16).digest()
        if h in vh:
            dup += 1; dup_tok += vh.pop(h)
print(f"val articles (>=16 tokens): {len(vh) + dup}; exact copies in train: {dup} ({dup_tok} tokens)", flush=True)

P = np.uint64(1_000_003)
pw = np.uint64(1)
for _ in range(a.n - 1):
    pw = pw * P


def ngram_hashes(d):
    d = np.asarray(d, dtype=np.uint64) + np.uint64(1)
    h = np.zeros(len(d) - a.n + 1, dtype=np.uint64)
    for k in range(a.n):                 # h = sum d[i + k] P^(n-1-k), uint64 arithmetic wraps (mod 2^64)
        h = h * P + d[k:len(d) - a.n + 1 + k]
    return h


vg = ngram_hashes(va)
eot_free = np.ones(len(vg), dtype=bool)
e = np.flatnonzero(np.asarray(va) == a.eot)
for x in e:                              # n-grams that span an article end are not counted
    eot_free[max(0, x - a.n + 1):x + 1] = False
vg_u = np.unique(vg[eot_free])
found = np.zeros(len(vg_u), dtype=bool)
for s in range(0, len(tr) - a.n + 1, a.chunk):
    th = ngram_hashes(tr[s:min(len(tr), s + a.chunk + a.n - 1)])
    found |= np.isin(vg_u, th)
    print(f"  scanned {min(len(tr), s + a.chunk) / 1e6:.0f}M train tokens: {found.mean() * 100:.2f}% of val {a.n}-grams seen", flush=True)
m = np.isin(vg[eot_free], vg_u[found])
print(f"val {a.n}-grams (within articles): {eot_free.sum():,}; occurring in train: {m.mean() * 100:.2f}% (upper bound)")
