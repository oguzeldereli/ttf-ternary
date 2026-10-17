> **Historical (to 25 Sep).** The current state is in [RUNS.md](RUNS.md) (headline), [RUN_INDEX.md](RUN_INDEX.md) and [FORMULAS.md](FORMULAS.md); this file keeps the systems result and the 25 Sep audit.

# Results

Two lines of work: a **systems** result (training a 27B native-ternary model on one
12 GB consumer GPU) and a **study** of master-free ternary training, where the
standard full-precision latent weights are removed.

> **The study section was re-run from scratch after an audit.** The previous version
> reported 1552 / 1214 / 707 perplexity for 0 / 2 / 3-bit variants. **Those numbers
> are invalid** — see §0. The systems numbers in §1 were unaffected.

---

## 0. Corrections

Two bugs in the training path invalidated every master-free number in the earlier
version of this file.

**No weight scale in the forward pass.** The kernel path multiplied activations by
raw trits {-1,0,1} with no scale, where BitNet b1.58 applies an absmean scale γ from
the latent weight (absent here). Measured consequences on the published checkpoints:

- each linear multiplied activation RMS by 19–75×
- attention logit std 260–700, attention entropy **0.008–0.018 nats**, mean max
  attention probability **0.99** — every head attended to exactly one token
- the stateless model's loss (7.62) was at the unigram entropy of the data (7.61):
  it had learned token frequencies and essentially nothing else

Fixed with a variance-preserving scale computed from the packed trits,
`β = 1/√(K·ρ)` where ρ is the nonzero-trit density (`trit_beta` in `bitnet/kernel.py`).
Output RMS becomes ≈ input RMS and fresh-model attention entropy goes to 4.6 nats
(near-uniform is 5.1).

**The float tail was bf16 with no master copy.** bf16 has 7 mantissa bits, so the
gap at 1.0 is 0.0078 while an Adam step is ~3e-4. All 17 RMSNorm gains were still
*exactly* 1.0 after 134k steps, and weight decay (lr·wd·w ≈ 6e-7) never did
anything. At the final LR ~70% of embedding updates rounded to zero.

**A third issue, memory rather than correctness:** RMSNorm upcasts to fp32 and was
called *outside* the gradient-checkpoint region, retaining two fp32 [B,T,d] tensors
per sublayer — 4.69 GiB at 32k tokens/step. Moving it inside cut peak VRAM from
7.67 to 2.70 GiB, which is what made the larger-batch runs fit.

Also fixed: the published bodies were **statistically independent of their init**
(66.6% of trits differed, exactly the value for two independent ternary draws at
ρ=0.69), the train/val split is article-level and leakage-free (4.80% 13-gram
overlap, no val article >90% covered), and the README repro commands omitted
`--grad_accum 1`, so they would not have reproduced the runs.

---

## 1. Systems: 27B ternary training on one 12 GB GPU

Native ternary ({-1,0,1}) transformer, weights packed **1.58-bit (5 trits/byte)**
resident during training. Custom Triton kernels keep the dense weight from ever
being materialized.

| stage | throughput @ seq512 | peak VRAM |
|-------|--------------------|-----------|
| unpack + `F.linear` (torch) | 67 tok/s | 10.86 GiB |
| + forward kernel | 92 tok/s | 10.26 GiB |
| **+ backward-`grad_x` + fused-flip kernels** | **172 tok/s** | **9.93 GiB** |

- 27B (`b27`) fits training in ~9.9 GiB (seq≤512); resident weights only 4.56 GiB.
- The wall is **compute, not memory**: 172 tok/s × ~540B Chinchilla tokens ≈ 100 yr.

### int8 tensor cores (new)

Activations are already 8-bit, so the forward is **exact in int32** where the bf16
path rounded partial sums. `mma.sync.aligned.m16n8k32...s8.s8.s32` confirmed in PTX.

Per-layer, M = 16,384 tokens (ms, speedup vs bf16):

| shape | fwd bf16 | fwd int8 | × | dx bf16 | dx int8 | × | dw dense | dw int8 | × |
|---|---|---|---|---|---|---|---|---|---|
| 110M 768→2048 | 1.13 | 0.66 | **1.70** | 1.59 | 4.52 | 0.35 | 0.74 | 5.43 | 0.14 |
| 110M 2048→768 | 1.29 | 0.66 | **1.97** | 1.33 | 2.23 | 0.60 | 0.74 | 2.33 | 0.32 |
| 27B 5120→13824 | 52.64 | 29.99 | **1.76** | 68.75 | 57.40 | 1.20 | 34.55 | 61.24 | 0.56 |
| 27B 13824→5120 | 63.14 | 33.54 | **1.88** | 82.15 | 49.38 | 1.66 | 36.09 | 40.31 | 0.90 |

End-to-end: **1.12×** at 110M (16,311 → 18,230 tok/s), **1.15×** at 27B (148 → 170
tok/s, peak 8.55 GiB with the memory fix). The GEMMs are only part of a step.

**Two optimization premises that did not survive measurement:**

1. *"The matmul is on CUDA cores, not tensor cores."* False — the bf16 kernels
   already emit `mma.sync.aligned.m16n8k16...bf16`. int8 dx is *slower* at 110M
   shapes (0.33–0.60×) because quantizing `gy` costs a full pass over [M,N]; it wins
   only at 27B shapes where the GEMM dominates that pass.
2. *"wgrad is the largest single cost, write an XNOR kernel for it."* False — dw is
   the **cheapest** of the three (0.74 ms vs 1.13 fwd, 1.59 dx). It is a dense GEMM
   on cuBLAS and nothing hand-written beat it: Triton int8 0.14–0.32×, cuBLASLt int8
   0.16–0.47×. `sign(dL/dy) ⊗ sign(x)` as an XNOR outer product reaches parity only
   at 27B (1.06–1.12×) and costs accuracy (71% sign agreement with dense dW, vs
   99.6% for int8 dw). Implemented as `--dw_mode sign`; not the default.

Under load the loop runs at 96–100% SM and 87–100% memory-controller utilization, so
there is no dataloader or elementwise stall to recover.

---

## 2. Study: what does removing the master weight cost?

**Setup, identical across every run:** 110M params (`small`: dim 768, 12 layers,
32k Llama vocab), seq 2048, **32,768 tokens per step**, 300M tokens, Wikipedia
20231101.en, cosine LR 3e-4→3e-5, seed 1337, shared data order, 1.0M fixed val
tokens. Only the weight-update rule changes.

| training rule | per-weight state | val ppl |
|---|---|---|
| latent master + STE + AdamW (**ceiling**) | fp32 latent + 2× fp32 Adam = 12 B | **15.79** |
| stateless flips, cosine-annealed rate | trit only (1.58 bit) | **98.56** |
| stateless flips, linear-annealed rate | trit only | 101.38 |
| stateless flips, cosine → 0.02% floor | trit only | 103.96 |
| stateless flips, constant rate (old recipe) | trit only | 135.84 |
| stateless flips, frozen absolute threshold | trit only | 160.68 |
| stateless flips, annealed + spatial error feedback (α=0.01) | trit only | 156.01 |
| stateless flips, annealed + per-weight flip lockout | trit only + 2 bits mask | 102.66 |

**Master-free ternary costs 6.2× perplexity** against its own ceiling at identical
config. That ratio is the number the previous version of this file could not report.

### The plateau is the flip rule, not gradient noise

See NOTES.md for the full argument. In short: gradient sign agreement between batch
halves is 65.7% (50% = noise), quadrupling the batch does not help, and the flip
rate is pinned at ~0.42% of weights per step regardless of batch, steps, tokens or
LR — because the threshold is normalized by the tensor's own mean |g|, which does
not shrink as the model converges. Annealing the rate to zero breaks the floor:
**135.84 → 98.56, a 27% cut, from one schedule.**

### Batch is not a substitute for temporal averaging

| tokens/step | steps | tokens | val ppl |
|---|---|---|---|
| 32,768 | 9,155 | 300M | 135.84 |
| 131,072 | 2,288 | 300M | 140.09 |
| 131,072 | 3,051 | 400M | 135.51 |

4× batch was worse at a fixed budget (fewer flip opportunities) and needed 33% more
tokens to draw level. It never got ahead.

### Does more data close the gap? Mostly no

Re-running the winning recipe at **600M tokens** (cosine anneal stretched over the
full budget, ~38 flips/weight vs ~19): **92.72**, against 98.56 at 300M.

2× the tokens and 2× the flip budget bought 5.8 ppl (5.9%), and the curve is
flattening — the last four checkpoints were 95.3 → 93.4 → 92.8 → 92.72. The gap to
the 15.79 ceiling stays **~5.9×**. The remaining gap is rule-bound, not data-bound.
(Caveat: this varies tokens *and* anneal length together, since the schedule must
still reach zero at the end.)

A related control: holding the flip rate at a non-zero floor (0.02%) instead of
letting it reach zero, with the same shape and budget, gives **103.96** vs 98.56.
The late gains come *from flipping stopping* — residual churn at convergence costs
about 5 ppl.

### 95-98% of flip work is undone — but stopping the churn by fiat does not help

Measured against the reproducible init: runs do 19-38 flips per weight and end ~0.9
levels from where they started (a trit can travel at most 2), so 95-98% of flips are
reversed by later flips. Letting each weight flip only once per N steps
(`--flip_lockout`) gives **102.66** vs 98.56 — slightly worse. A flip's direction is
a noisy sample (65.7% sign agreement), so locking freezes in the ~1/3 that were wrong
rather than making flips count, and it also cuts the flip budget. See RUNS.md for the
screens and for why short mid-training evals could not rank the variants.

### Spatial error feedback makes it worse

Pushing each layer's unapplied flip residual into `grad_x` — error feedback with no
stored state — gives **156.01** against 98.56 at the smallest alpha tried in the full
run (0.01), and 600-step probes show the damage growing steeply with alpha (255.6 /
273.7 / 446.0 / 828.1 at α = 0 / 0.01 / 0.03 / 0.1). The residual is zero-mean
flip-sampling noise (`E[fired] = p`), so feeding it upstream adds noise to every
earlier layer's gradient. The full run blew up while flipping was heavy (ppl 204 →
458) and only recovered as the anneal switched flipping off. See RUNS.md.

---

## 3. Negative result: superposition can't shrink the accumulator

(Unchanged; measured on the 3-bit evidence variant, which is the *accumulator*
branch of the study rather than the stateless one.)

- **Evidence is 75% dense** (19.3M of 25.7M counters nonzero at once).
- Superposition crosstalk is `√(K/d)`. At d=2²⁰, load = 18 → sign-recovery **56%**
  (coin flip), and 95% of zero-weights get phantom evidence.
- To recover even 91% you need d=2²⁶ = **67 MB**, vs the exact 3-bit array at
  **9.6 MB**. At the array's memory budget the sketch recovers at **52%**.
- **Entropy floor:** 2.66 bits/weight → 8.5 MB incompressible; the 3-bit array
  (9.6 MB) is already ~13% above it.

This applies to *any* scheme that compresses per-weight state below its entropy,
including per-neuron or row/column accumulators.

---

## 4. Honest limitations

- **n = 1.** No seed repeats. Evals are 1.0M tokens and wobble ±5 ppl, so the
  differences *within* the annealed group (98.56 / 101.38 / 103.96) are **not**
  separated by this data. The large gaps (annealed vs constant vs frozen-threshold,
  and all of them vs the ceiling) are far outside that noise.
- **One model size, one dataset.** Everything is 110M on Wikipedia.
- **Everything is undertrained.** 300M tokens for a 110M model is ~7% of Chinchilla;
  both the baseline and the ternary runs were still descending. The 6.2× is a
  snapshot at this budget, not a converged asymptote.
- **No tuning parity.** One LR for the baseline, one flip rate for the arms; neither
  side was swept.
- **The bit-depth question is unmeasured.** The 2-bit / 3-bit evidence runs were not
  re-run after the β fix, so the accumulator-depth curve — the original headline of
  this study — currently has no valid data points.
- The baseline run was stopped and resumed once; its data sampler re-seeds on
  resume, so it is not a bit-exact single run.

## Reproduce

```bash
python3 prep_wiki.py 4e8 --tokenizer llama --out_prefix data/wiki32k
# stateless, annealed flip rate (the best master-free result)
python3 -m bitnet.train --preset small --mode kernel --int8 --dw_mode dense \
  --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 \
  --lr 3e-4 --min_lr 3e-5 --rate_schedule cosine --track_flips --out_dir checkpoints/anneal
# the ceiling
python3 -m bitnet.train --preset small --mode master --master_dtype fp32 ... \
  --lr 1.5e-3 --min_lr 1.5e-4 --out_dir checkpoints/baseline
python3 diag_snr.py --ckpt checkpoints/anneal/ckpt.pt --seq 2048   # gradient SNR
python -m scripts.plots.plot_compare checkpoints/anneal checkpoints/baseline    # plots
```
