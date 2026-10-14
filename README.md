# ternary-LLM

Ternary ({-1, 0, 1}) language models trained **without full-precision master weights**, and pretrained models
converted to ternary weights with the same update. The weights stay packed at 1.6 bits (5 trits per byte) during
training; the optimizer state is low-rank, so it grows with the layers' widths rather than with the number of weights.

The method is **two-timescale threshold flips (TTF)**, `--ts`. The Adam-normalised gradient, with a factored second
moment, feeds a leaky accumulator kept at low rank. A weight moves one level when its accumulated evidence crosses a
threshold, and the threshold is then spent from the accumulator. The trits are the only per-weight state. TTF came
out of an ablation of master-weight training, which showed what master weights provide: an integrator of normalised
evidence with a memory of about 1000 steps, and immediate moves when it crosses a boundary; neither a full-rank first
moment nor a per-weight second moment.

The paper (local draft, `docs/paper/ttf_paper.tex`, build with `scripts/paper/build_pdf.sh`) has every result below
with its run; `docs/paper` Appendix "Index of mechanisms and results" lists everything that was tried.

## Results

All numbers are validation loss (nats per token) on fixed windows: all 974 non-overlapping 2048-token windows of the
Wikipedia validation set and 2000 windows of FineWeb-Edu, which was never used for any choice
(`scripts/eval/final_eval.py`). "Master + extras" is the BitNet b1.58 straight-through recipe with exactly the same
auxiliary float parameters as TTF (row / column scales, rank-16 float adapter, attention temperature): the fair
reference.

**110M, 300M tokens of Wikipedia (Llama-2 32k tokenizer)**

| method | Wikipedia | FineWeb-Edu |
|---|---|---|
| **TTF, rank 512 (final recipe)** | **2.6728** (seed 2: 2.6671) | **3.6226** (3.6168) |
| TTF, full rank | 2.6480 | 3.5976 |
| TTF, rank 512, no float extras | 2.6907 | 3.6435 |
| master + extras | 2.7166 | 3.6830 |
| master (BitNet b1.58) | 2.736 | 3.707 |
| low-rank Adam STE, rank 512 | 2.8867 | 3.8678 |

At 600M tokens of FineWeb-Edu TTF leads master + extras by 0.056 (3.0423 vs 3.0982).

**Scale (fixed windows, Wikipedia / FineWeb-Edu; difference to master + extras at its best learning rate)**

| size | TTF state (numbers per weight) | TTF − master |
|---|---|---|
| 340M | rank 512 (0.79) / full rank | −0.032 / −0.071 |
| 1.3B | rank 1024 (0.79), two seeds | −0.010 / −0.027 |
| 2.9B (FineWeb-Edu) | rank 1536 (0.79) | +0.002 / −0.004 |

TTF's gap to master is a smooth function of the size of its state and closes at 0.57–0.70 numbers of state per weight
at every size (about one byte per weight at the 12 bits the accumulator tolerates, against 12 bytes for master).

**Converting pretrained models** (`--init_hf`): trits chosen by GPTQ onto the ternary grid, frozen while the float
parts calibrate, then moved by TTF at a threshold four times the from-scratch one (a smaller threshold triggers an
avalanche of harmful moves after a conversion).

| model, recovery tokens | full precision | converted, before training | after |
|---|---|---|---|
| TinyLlama-1.1B, 50M | 1.875 | 8.01 | **2.2320** (latent-weight conversion: 2.5973) |
| Qwen3.5-0.8B-Base, 50M | 2.556 | 4.96 | 2.8503 |
| Qwen3.8-27B, 2M (probe) | 1.9034 | 3.8507 | 2.3644 (8 windows) |

Qwen3.8-27B is converted layer by layer (streaming, about 20 minutes) and trained on one 24 GB GPU (peak 21.2 GiB);
the converted model runs on a 12 GB laptop GPU at 22.7 tokens per second (`scripts/infer/ttf_qwen.py`).

## Memory

- A 23.0B-parameter model (22.85B ternary) trains on one 16 GB GPU (11.1–11.6 GiB at ranks 256–1024, state in host memory with
  asynchronous copies; `scripts/lab/b27_lean.sh`). At its leanest it peaks at 9.73 GiB.
- **Micro-batches without a dense gradient** (`--ts_micro`): each micro-batch's gradient enters the state and is
  discarded; 4 × 4 micro-batches match one pass of batch 16 over the full schedule (2.6733 vs 2.6728) at 3.1 instead of
  7.4 GiB.
- **State formats:** a 12-bit accumulator (`--ts_store_a fp12s`: per-column scale, sign, 5 exponent and 6 mantissa
  bits, two values in 3 bytes) matches 32 bits; at fixed memory a higher rank in 12 bits beats a lower rank in 32
  (rank 340 at 12 bits 2.7086 vs rank 128 at 32 bits 2.8410).

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/data/prep_wiki.py 4e8 --tokenizer llama --out_prefix data/wiki32k     # Wikipedia, 32k vocab
python3 -m scripts.data.prep_fineweb 2e9 --out_prefix data/fwedu32k                    # FineWeb-Edu (optional)

# TTF, 110M, 300M tokens (the final recipe)
python -m bitnet.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --steps 9155 --warmup 305 --int8 --dw_mode dense --tail_fp32 \
    --lr 1.5e-3 --min_lr 1.5e-4 --lookahead 0 --ckpt_skip 2 --eval_interval 250 --eval_iters 30 \
    --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp \
    --ts --ts_nomom --ts_fused --gemm int8mm --lowrank 512 --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 \
    --ts_anneal 0.75 --ts_tau_anneal --out_dir checkpoints/ttf

# master-weights reference with the same float extras
python -m bitnet.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --steps 9155 --warmup 305 --lr 1.5e-3 \
    --min_lr 1.5e-4 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --out_dir checkpoints/master

# fixed-window evaluation, paired comparison, zero-shot
python scripts/eval/final_eval.py checkpoints/ttf checkpoints/master
python scripts/eval/compare.py checkpoints/ttf checkpoints/master --set fw
python3 scripts/eval/zeroshot_tok.py && python scripts/eval/zeroshot.py checkpoints/ttf

# convert a pretrained Llama-architecture model (the recipe; scripts/quant/recover_run.sh has the full command)
python -m bitnet.train --init_hf PATH/TO/TinyLlama --init_hf_method gptq --ts_start 400 --ts_theta 64 ...
```

Add `--resume` to continue any run.

## Main flags

| flag | what |
|---|---|
| `--ts`, `--lowrank R` | the method; accumulator rank |
| `--ts_nomom` | no short momentum: the normalised gradient enters the accumulator (the final recipe; `--ts_rank_s` then unused) |
| `--ts_theta`, `--ts_tau`, `--ts_anneal P`, `--ts_tau_anneal` | threshold, leak, step scale ρ^P (ρ = lr / peak lr), leak scaled with the step |
| `--ts_fused` | update each layer inside its backward pass (one layer's gradient at a time) |
| `--ts_micro` | with `--grad_accum M`: the micro-batch update, no dense gradient |
| `--ts_store_m`, `--ts_store_a int8 / fp16 / fp12s` | real compressed state |
| `--ts_offload_async`, `--ts_offload_keep F` | state in pinned host memory (all, or all but the first share F of layers), asynchronous copies |
| `--ts_rank_adapt E` | per-layer ranks reallocated every E steps under the fixed total |
| `--init_hf DIR`, `--init_hf_method als / gptq`, `--init_stream` | convert a pretrained model (Llama, Qwen3.5 / 3.8); `--init_stream` layer by layer |
| `--ts_start S` | trits frozen until step S (conversion) |
| `--tern_emb tied / untied / tied_wide:M` | ternary token embedding (and head), trained by TTF |
| `--moe E:K[:H]` | mixture of ternary experts, each with its own state |

## Layout

```
bitnet/          model, Triton kernels, TTF and other update rules, training loop
  train.py       training entry point (python -m bitnet.train --help); ts_layer = the TTF update
  kernel.py      packed-ternary kernels (int8 / bf16 GEMMs, input gradient, packing)
  flip.py        ternary layers, gradient capture / fused hooks
  hf_init.py     conversion of pretrained Llama models (ALS, GPTQ onto the ternary grid, accumulator warm start)
  qwen.py        Qwen3.5 / 3.8 (Gated DeltaNet + attention) with ternary linears; streaming conversion
  tern_emb.py    ternary embeddings; movediag.py: move diagnostics
  master.py      master-weights baseline (BitNet b1.58 STE + AdamW); master_opt.py: its ablations
  model.py, config.py   Llama-style transformer, presets (small = 110M, d1024_l24 = 340M, d2048_l24 = 1.3B, b27)
scripts/
  infer/         ttf_run.py (our models), ttf_qwen.py + ttf_kernels.py (converted Qwen, 2-bit dp4a decode)
  quant/         conversion launchers, Qwen checks, data for chat models (chat_data.py, sft_mix.py)
  eval/          final_eval.py, compare.py, zeroshot*.py
  analysis/      memory_27b.py, rank_spectrum.py, val_overlap.py, diagnostics
  lab/, remote/, myriad/, night/   launchers for the lab PCs, the 4090 workstation, UCL Myriad (A100)
  data/, plots/, paper/  data preparation, figures, the paper's build
tests/           test_kernel_ref.py (kernels vs dense references) and sanity tests
docs/            see below
```

## Docs

- `docs/paper/ttf_paper.tex`: the paper (all results, with the index of mechanisms).
- [docs/QUEUE.md](docs/QUEUE.md): runs in progress and the dated log of results.
- [docs/TODO.md](docs/TODO.md): open questions, limitations, paper to-do.
- [docs/RUNS.md](docs/RUNS.md) / [docs/RUN_INDEX.md](docs/RUN_INDEX.md): lab notebook and run index (to early October).
- [docs/FORMULAS.md](docs/FORMULAS.md): update rules; [docs/THREATS.md](docs/THREATS.md): objections and evidence.
- `docs/RESULTS.md`, `docs/NOTES.md` and the `MORNING_SUMMARY_*` files are historical.
