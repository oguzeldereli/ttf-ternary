# ternary-LLM

Native ternary ({-1, 0, 1}) language models trained **without full-precision master weights**. The weights stay
packed at 1.6 bits (5 trits per byte) during training. The optimizer state is low-rank, so its size grows like the
layer width rather than the number of weights.

The current method is **two-timescale threshold flips (TTF)**, `--ts`. A short low-rank momentum gives the direction.
It is normalised like Adam and integrated by a long low-rank accumulator. A weight moves one level when its
accumulated evidence crosses a threshold, and the move is then subtracted from the accumulator. It came out of an
ablation of master-weight training, which showed what master weights actually provide: a short momentum, a per-weight
integrator with a memory of about 1000 steps, and immediate moves.

## Results

110M parameters, 300M tokens of English Wikipedia (Llama-2 32k tokenizer). All models are scored on the same fixed
windows of each validation set (`scripts/eval/final_eval.py`). The rebuilt models reproduce their logged losses to
within 6e-5. FineWeb-Edu was never used for any choice.

| model (seeds) | optimizer state | Wikipedia val loss | FineWeb-Edu val loss |
|---|---|---|---|
| **TTF, ranks 512 (3)** | low-rank, ~8 B per ternary weight at this width | **2.6888** | **3.6453** |
| master weights + the same float extras (2) | 16 B per weight | 2.7166 | 3.6830 |
| master weights, BitNet b1.58 recipe (3) | 16 B per weight | 2.7359 | 3.7071 |
| our earlier flip rule | low-rank | 2.7973 | 3.7545 |
| full-precision fp32 (reference) | 16 B per weight | 2.683 (training-time eval) | |

- **Against master with the same float extras:** TTF is −0.028 on Wikipedia, 95% interval [−0.029, −0.027] from a
  paired block bootstrap over windows. On FineWeb it is −0.038. Seed spread is 0.002–0.0035, so the gap is more than
  10× the seed noise.
- **Master learning rate:** of 7.5e-4 / 1.5e-3 / 3e-3 / 6e-3, the 1.5e-3 used for master is the best.
- **1.3B** (fixed windows, one seed each): TTF 2.5510 vs master 2.5706 (−0.020 [−0.021, −0.018]); FineWeb-Edu
  3.5009 vs 3.5489 (−0.048).
- **340M** (fixed windows, one seed each, seed 2 running): TTF 2.5774 vs master + the same float extras 2.6061
  (−0.029 [−0.030, −0.027]) and master 2.6461; FineWeb-Edu 3.5251 vs 3.5654 / 3.6167. With the older step scaling
  (p = 1) TTF was level with master + extras. Master's learning rate was tuned at 110M; at 340M lr 1e-3 is better
  mid-run (level with TTF at step 4000), and its final decides this comparison.
- **FineWeb-Edu, 600M tokens** (110M): TTF 3.0861 vs master 3.1508.
- **Zero-shot at 110M:** LAMBADA 0.135 vs 0.123 for master + extras. The multiple-choice tasks are at chance at this
  scale.

## Memory

- **Measured peaks of lean TTF training** (micro-batch 1–2 × 2048): 110M 0.38–0.63 GiB, 340M 0.68–1.35 GiB, 1.3B
  1.61–2.97 GiB (state on host / on GPU), ~10B 12.3 GiB, ~15B 17.0 GiB.
- **27B-class model on one 16 GB GPU, measured.** The 26.5B-parameter preset `b27` (22.85B ternary weights) trains
  with TTF at micro-batch 1 × 2048 with the update inside the backward pass, the lean float tail, the state in exactly
  sized page-locked host memory with asynchronous copies (`--ts_offload_async`) and full activation checkpointing
  (`scripts/lab/b27_lean.sh`):

  | ranks | state format | GPU peak (PyTorch) | host memory |
  |---|---|---|---|
  | 256 | int8 / fp16 | 11.1 GiB | 11.2 GiB |
  | 512 | int8 / fp16 | 11.1 GiB | 17.0 GiB |
  | 512 | int8 / 12-bit | 11.5 GiB | 15.4 GiB |
  | 1024 | int8 / 12-bit | 11.6 GiB | 23.7 GiB |

  Ranks 256 / 512 were measured on an RTX 4070 Ti Super (16 GB) with 30 GB RAM, ranks 1024 on a 3090 Ti with 125 GB RAM
  (GPU memory is the same allocation on either card). With the state on the host the GPU peak barely depends on the
  rank; host memory is the limit. A ~15B model (width 6144) peaks at 9.4 GiB on the 16 GB card.
- **Host offload costs no time with `--ts_offload_async`** (copies on side streams, next layer prefetched): 1.3B at
  micro-batch 1 2.80 s/step vs 2.79 with the state on the GPU. torch's pinned allocator rounds every buffer up to a
  power of two (at 27B ranks 1024: 29.7 GiB for 21.3 GiB of state); the state is allocated exactly instead.
- **State formats.** The momentum (direction) can be stored in 8 bits with a per-column scale (int8 or fp8) at no
  cost; 4 bits fails. The accumulator needs about 5 mantissa bits: 5 bits +0.002, 4 bits +0.05, 3 bits +0.15, int8
  +0.01–0.02. A real 12-bit accumulator (`--ts_store_a fp12s`: per-column scale, sign, 5 exponent and 6 mantissa bits,
  two values in 3 bytes) is level with 16 bits so far (quality run running). The momentum rank cannot be cut below the
  accumulator rank (+0.05 for rank 256 or a per-type split under an accumulator of 512).
- **Size.** Measured at 110M: int8 + fp16 cuts the state from 685 to 258 MiB with no change in loss. At 27B the
  state is 2 / 4 / 8 bits per weight at ranks 256 / 512 / 1024, against 128 bits for master weights + AdamW
  (`scripts/analysis/memory_27b.py`).

## Evaluation and checks

- `docs/THREATS.md` lists every objection we could think of, the evidence against each, and what is still open.
- No validation article appears in the training data. At most 1.6% of validation 32-grams occur anywhere in it
  (`scripts/analysis/val_overlap.py`).
- `tests/test_kernel_ref.py` checks the ternary kernels against dense references: 69 checks, maximum error at bf16
  rounding.
- The master baseline is the BitNet b1.58 straight-through recipe. It is given exactly the same float extras as TTF
  in the "master + extras" reference.
- A ternary Bop baseline (latent-free flips with one timescale, `--bop`) is 0.38 behind TTF.

## Method in one screen

Per ternary layer and step, with ρ = lr / peak lr. Full details in [docs/FORMULAS.md](docs/FORMULAS.md) §2a.

- **v:** factored second moment, one statistic per row and one per column (β₂ 0.95).
- **m:** momentum of rank r_s (β₁ 0.9), kept by one subspace-iteration step per step.
  The normalised step is u = −m̂ / √v̂.
- **A:** accumulator of rank r. A ← (1 − λ/τ)·A + ρ^p·u, with τ ≈ 1000 and p = 0.5.
- **Move:** a weight moves one level toward sign(A) where |A| ≥ θ (θ = 16) and the move stays in {−1, 0, 1}.
  The move is then spent from the accumulator: U ← U − θ·D·V, with D the moves made.

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/data/prep_wiki.py 4e8 --tokenizer llama --out_prefix data/wiki32k     # Wikipedia, 32k vocab
python3 -m scripts.data.prep_fineweb 2e9 --out_prefix data/fwedu32k                    # FineWeb-Edu (optional)

# TTF, 110M, 300M tokens (the headline run)
python -m bitnet.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 \
    --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --lookahead 0 \
    --ckpt_skip 2 --eval_interval 250 --eval_iters 30 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp \
    --lowrank 512 --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5 --out_dir checkpoints/ttf

# master-weights reference (same data, batch, schedule)
python -m bitnet.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --steps 9155 --warmup 305 --lr 1.5e-3 \
    --min_lr 1.5e-4 --out_dir checkpoints/master

# final evaluation on fixed windows, paired comparison, zero-shot
python scripts/eval/final_eval.py checkpoints/ttf checkpoints/master
python scripts/eval/compare.py checkpoints/ttf checkpoints/master --set fw
python3 scripts/eval/zeroshot_tok.py && python scripts/eval/zeroshot.py checkpoints/ttf
```

Add `--resume` to continue any run. `--ts_fused` runs the update inside the backward pass, the same maths with less
memory.

## TTF flags

| flag | what |
|---|---|
| `--ts`, `--lowrank R`, `--ts_rank_s R_s` | the method; accumulator rank and momentum rank |
| `--ts_theta`, `--ts_tau`, `--ts_anneal P`, `--ts_tau_anneal` | threshold, leak, step scaling ρ^P, leak scaled with the schedule |
| `--ts_fused` | update each layer inside its backward pass (only one layer's gradient exists at a time) |
| `--ts_store_m int8 --ts_store_a fp16` | real compressed state: codes + per-column scales, or 16-bit |
| `--ts_offload` | state in pinned host memory, copied per layer for its update |
| `--ts_orth qr / chol64` | orthonormalisation: Householder (default), or two-pass Cholesky-QR in fp64. Do not use `chol`: its fp32 Gram matrix can lock subspaces at zero |
| `--ts_rank_kv R` | separate rank for the attention k / v projections |
| `--ts_diag N`, `--ts_shadow` | diagnostics: what the rank-r projections drop, and a full-rank shadow accumulator |
| `--gemm cublas`, `--int8` | matrix-product path; int8 tensor-core forward |
| `--bop`, `--bop_tau`, `--bop_gamma` | the ternary Bop baseline (comparison only) |

## Layout

```
bitnet/          model, Triton kernels, TTF and other update rules, training loop
  train.py       training entry point (python -m bitnet.train --help); ts_layer = the TTF update
  kernel.py      packed-ternary kernels (int8 / bf16 GEMMs, input gradient, packing)
  flip.py        ternary layers, gradient capture / fused hooks
  master.py      master-weights baseline (BitNet b1.58 STE + AdamW); master_opt.py: its ablations
  model.py, config.py   Llama-style transformer, presets (small = 110M, d1024_l24 = 340M, d2048_l24 = 1.3B, b27)
scripts/
  eval/          final_eval.py, compare.py, zeroshot*.py, rescue.sh (copy final checkpoints)
  analysis/      memory_27b.py, val_overlap.py, audit_numbers.py, diagnostics
  lab/, remote/, myriad/, night/   launchers for the lab PCs, the 4090 workstation, UCL Myriad (A100)
  data/, plots/  data preparation, figures (docs/figures)
tests/           test_kernel_ref.py (kernels vs dense references) and sanity tests
docs/            see below
```

## Docs

- [docs/RUNS.md](docs/RUNS.md): lab notebook, newest at the bottom.
- [docs/RUN_INDEX.md](docs/RUN_INDEX.md): every run with its result.
- [docs/FORMULAS.md](docs/FORMULAS.md): every update rule.
- [docs/THREATS.md](docs/THREATS.md): objections and the evidence.
- [docs/QUEUE.md](docs/QUEUE.md): what is running.
- [docs/figures/](docs/figures/README.md): figures.
- `docs/overview/index.html`: overview page.
- `docs/RESULTS.md` and `docs/NOTES.md` are historical (to 25 Sep).

## Open questions

- **Late-training slope.** With steps × ρ^0.5 (p = 0.5) the 340M lead holds to the end (with p = 1 master + extras
  caught up). At 1.3B (p = 0.5) the lead shrank to −0.008 and widened again to −0.020 at the end.
- **Rank vs width.** Ranks 512 match or beat master up to 1.3B (rank / width 0.25). Low rank fails in the accumulator,
  whose energy is spread over many directions. At batch 16 the share that ranks 512 captures falls with width (0.96 /
  0.91 / 0.77 at 110M / 340M / 1.3B), while energy capture overstates the need (1.3B ranks 512 still beats master). A
  width sweep at batch 2 (up to 15B) cannot settle it: at that batch the gradient is mostly noise and a lower rank
  filters it. The deciding run is 1.3B at ranks 256 (running).
- **Wall-clock.** On an A100 at 1.3B, a TTF step is about 5 s against 4.5 s for master. The QR timing is being
  re-measured.
- Token budgets are small (300M–600M), and the 340M and 1.3B comparisons have one seed each.
