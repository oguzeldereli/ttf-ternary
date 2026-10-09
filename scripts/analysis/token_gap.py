"""Where is the gap between two runs, token by token? Paired comparison of two scripts/analysis/token_dump.py outputs on
the same windows (A minus B; A = TTF, B = master). Splits the mean per-token gap, with a bootstrap standard error over
windows, by:
  - training-set frequency of the target token (deciles of the evaluated targets);
  - context position (0-1, 2-15, 16-127, 128-511, 512-2047);
  - difficulty: the two runs' mean loss on the token (quintiles; B's loss alone would bias the split by regression
    to the mean);
  - in-context repetition: the target token / the (previous, target) pair already occurred earlier in the window
    (copyable, induction) or not;
  - word start vs continuation (Llama 32k tokenizer table);
and lists the tokens that carry the most of the summed gap. "share" = that bucket's part of the total summed gap.
A gap spread evenly over every split (each share ~ the bucket's share of tokens) points at a global cause (noise floor
of the step size); a gap concentrated on rare tokens / long context / copies points at capacity or a specific mechanism.
    python scripts/analysis/token_gap.py A.npz B.npz [--train data/fwedu32k_train.bin] [--label NAME]
"""
import argparse, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--train", default="data/fwedu32k_train.bin"); ap.add_argument("--label", default="")
args = ap.parse_args()
A, B = np.load(args.a), np.load(args.b)
assert (A["tok"] == B["tok"]).all(), "different windows"
tok = A["tok"].astype(np.int64); la = A["loss"].astype(np.float32); lb = B["loss"].astype(np.float32)
tgt, prev = tok[:, 1:], tok[:, :-1]
d = la - lb
n, L = d.shape
rng = np.random.default_rng(0)
BOOT = rng.integers(0, n, (300, n))


def stat(mask):
    """mean gap on the masked tokens, bootstrap SE over windows, share of the total summed gap, share of tokens"""
    s = (d * mask).sum(1); c = mask.sum(1)
    m = s.sum() / max(c.sum(), 1)
    bs = s[BOOT].sum(1) / np.maximum(c[BOOT].sum(1), 1)
    return m, bs.std(), s.sum() / d.sum(), c.sum() / mask.size


def table(title, masks, labels):
    print(f"\n{title}")
    print(f"  {'bucket':>14s} {'tokens %':>9s} {'gap':>8s} {'+-':>6s} {'gap share %':>11s}   B loss   A loss")
    for m, lab in zip(masks, labels):
        g, se, sh, ts = stat(m)
        print(f"  {lab:>14s} {100 * ts:9.1f} {g:+8.4f} {se:6.4f} {100 * sh:11.1f}   {lb[m].mean():6.3f}   {la[m].mean():6.3f}")


cache = os.path.splitext(args.train)[0] + "_unigram.npy"
if os.path.exists(cache):
    uni = np.load(cache)
else:
    tr = np.memmap(args.train, dtype=np.uint16, mode="r"); uni = np.zeros(65536, np.int64)
    for s in range(0, len(tr), 100_000_000):
        uni += np.bincount(np.asarray(tr[s:s + 100_000_000]), minlength=65536)
    np.save(cache, uni)
print(f"{args.label}  A {args.a} (step {int(A['step'])})  vs  B {args.b} (step {int(B['step'])})")
print(f"windows {n} x {L}; mean loss A {la.mean():.4f}  B {lb.mean():.4f}  gap {d.mean():+.4f} "
      f"(+-{d.mean(1)[BOOT].mean(1).std():.4f})")

f = uni[tgt]
q = np.quantile(f, np.linspace(0, 1, 11))
dec = np.clip(np.searchsorted(q[1:-1], f, side="right"), 0, 9)
table("by training frequency of the target (decile 0 = rarest)", [dec == k for k in range(10)],
      [f"d{k} ({int(q[k])}-)" if k else f"d0 (<{int(q[1])})" for k in range(10)])

pos = np.broadcast_to(np.arange(L), d.shape)
PB = [(0, 2), (2, 16), (16, 128), (128, 512), (512, 2048)]
table("by context position", [(pos >= a) & (pos < b) for a, b in PB], [f"{a}-{b - 1}" for a, b in PB])

lm = 0.5 * (la + lb)
qb = np.quantile(lm, np.linspace(0, 1, 6))
k5 = np.clip(np.searchsorted(qb[1:-1], lm, side="right"), 0, 4)
table("by difficulty (quintile of the mean loss of A and B; 0 = easiest)", [k5 == k for k in range(5)],
      [f"q{k} ({qb[k]:.2f}-)" for k in range(5)])

seen = np.zeros_like(tgt, dtype=bool); pair = np.zeros_like(tgt, dtype=bool)
for i in range(n):          # did the target (and the pair prev -> target) already occur earlier in this window?
    t, p = tgt[i], prev[i]
    first = {}
    for j, x in enumerate(t):
        if x in first: seen[i, j] = True
        else: first[x] = j
    key = p * 65536 + t
    _, idx = np.unique(key, return_index=True)
    fm = np.ones(L, bool); fm[idx] = False
    pair[i] = fm
table("in-context repetition", [~seen, seen & ~pair, pair],
      ["new token", "token seen", "pair seen (copy)"])

ws_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordstart_llama32k.npy")
ws = np.load(ws_path)
st = ws[np.clip(tgt, 0, len(ws) - 1)]
table("word start vs continuation", [st, ~st], ["word start", "continuation"])

tot = np.bincount(tgt.ravel(), weights=d.ravel(), minlength=65536)
cnt = np.bincount(tgt.ravel(), minlength=65536)
try:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    name = lambda i: repr(tk.convert_ids_to_tokens(int(i)))
except Exception:
    name = lambda i: str(int(i))
for title, order in (("tokens adding the most to the gap (A worse)", np.argsort(-tot)),
                     ("tokens where A gains the most (A better)", np.argsort(tot))):
    print(f"\n{title}: token, count, mean gap, share of total %")
    for i in order[:15]:
        print(f"  {name(i):>16s} {cnt[i]:7d} {tot[i] / max(cnt[i], 1):+.4f} {100 * tot[i] / d.sum():6.1f}")
