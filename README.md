# ttf-ternary

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23394608.svg)](https://doi.org/10.5281/zenodo.23394608)

Ternary ({-1, 0, 1}) language models trained **without full-precision master weights**, and pretrained models
converted to ternary weights with the same update. The weights stay packed at 1.6 bits (5 trits per byte) during
training; the optimizer state is low-rank, so it grows with the layers' widths rather than with the number of weights.

The method is **two-timescale threshold flips (TTF)**, `--mode ttf` (the default). The Adam-normalised gradient, with a factored second
moment, feeds a leaky accumulator kept at low rank. A weight moves one level when its accumulated evidence crosses a
threshold, and the threshold is then spent from the accumulator. The trits are the only per-weight state. TTF came
out of an ablation of master-weight training, which showed what master weights provide: an integrator of normalised
evidence with a memory of about 1000 steps, and immediate moves when it crosses a boundary; neither a full-rank first
moment nor a per-weight second moment.

The paper (preprint to follow) has every result below with its run, and an appendix indexing every mechanism that was
tried with its result.

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

**Scale (fixed windows; TTF minus master + extras at its best learning rate)**

| size | TTF state (numbers per weight) | Wikipedia | FineWeb-Edu |
|---|---|---|---|
| 340M | rank 512 (0.79) | −0.032 | −0.042 |
| 340M | full rank 1024 (1.57) | −0.071 | −0.087 |
| 1.3B | rank 1024 (0.79), two seeds | −0.010 | −0.027 |
| 1.3B | full rank 2048 (1.57) | −0.051 | −0.069 |
| 2.9B (trained on FineWeb-Edu) | rank 1536 (0.79) | +0.002 | −0.004 |

TTF's gap to master is a smooth function of the size of its state and closes at 0.57–0.70 numbers of state per weight
at every size (about one byte per weight at the 12 bits the accumulator tolerates, against 12 bytes for master).

**Converting pretrained models** (`--init-hf`): trits chosen by GPTQ onto the ternary grid, frozen while the float
parts calibrate, then moved by TTF at a threshold four times the from-scratch one (a smaller threshold triggers an
avalanche of harmful moves after a conversion).

| model, recovery tokens | full precision | converted, before training | after |
|---|---|---|---|
| TinyLlama-1.1B, 50M | 1.875 | 8.01 | **2.2320**; with distillation 2.1878 (latent-weight conversion: 2.5973) |
| Qwen3.5-0.8B-Base, 50M | 2.556 | 4.96 | 2.8503 (distillation does not help at this size) |
| Qwen3.8-27B, 2M (probe) | 1.9034 | 3.8507 | 2.3644 (8 windows) |

Qwen3.8-27B is converted layer by layer (streaming, about 20 minutes) and trained on one 24 GB GPU (peak 21.2 GiB);
the converted model runs on a 12 GB laptop GPU at 22.7 tokens per second (`scripts/infer/ttf_qwen.py`). The 27B row
is an early 60-step probe; a release model (about 40M tokens of recovery with distillation) is in training.

## Memory

- A 23.0B-parameter model (22.85B ternary) trains on one 16 GB GPU (11.1–11.6 GiB at ranks 256–1024, state in host memory with
  asynchronous copies, `--ttf-offload async`; measured with `scripts/lab/b27_lean.sh` at tag `pre-consolidation-2026-10-17`).
  At its leanest it peaks at 9.73 GiB.
- **Qwen3.8-27B on one 24 GB GPU** (RTX 3090 Ti): converted layer by layer and trained at 32k tokens per step (2 × 8 ×
  2048) in 68.9 s per step, peak 19.7 GB (`recipes/convert-27b.toml`; the whole pipeline: [docs/CONVERT_27B.md](docs/CONVERT_27B.md)).
- **Micro-batches without a dense gradient** (`--ttf-accum micro`, the default): each micro-batch's gradient enters the state and is
  discarded; 4 × 4 micro-batches match one pass of batch 16 over the full schedule (2.6733 vs 2.6728) at 3.1 instead of
  7.4 GiB.
- **State formats:** a 12-bit accumulator (`--ttf-store fp12s`: per-column scale, sign, 5 exponent and 6 mantissa
  bits, two values in 3 bytes) matches 32 bits; at fixed memory a higher rank in 12 bits beats a lower rank in 32
  (rank 340 at 12 bits 2.7086 vs rank 128 at 32 bits 2.8410).

## Quickstart

```bash
pip install -e ".[data]"          # ".[qwen]" for Qwen3.5 / 3.8, ".[test]" for pytest; the ttf-train command = python -m ttf.train
python3 scripts/data/prep_wiki.py 4e8 --tokenizer llama --out_prefix data/wiki32k     # Wikipedia, 32k vocab
python3 -m scripts.data.prep_fineweb 2e9 --out_prefix data/fwedu32k                    # FineWeb-Edu (optional)

# TTF, 110M, 300M tokens (the final recipe: recipes/scratch-110m.toml)
python -m ttf.train --recipe scratch-110m --out-dir runs/ttf

# BitNet b1.58 (master weights + STE) with the same float extras, and the full-precision model
python -m ttf.train --recipe scratch-110m --mode bitnet --out-dir runs/bitnet
python -m ttf.train --recipe scratch-110m --mode fp32 --out-dir runs/fp32

# fixed-window evaluation, paired comparison, zero-shot
python scripts/eval/final_eval.py runs/ttf runs/bitnet
python scripts/eval/compare.py runs/ttf runs/bitnet --set fw
python3 scripts/eval/zeroshot_tok.py && python scripts/eval/zeroshot.py runs/ttf

# convert a pretrained model (Llama architecture or Qwen3.5 / 3.8)
python -m ttf.train --recipe convert-110m --out-dir runs/convert
python -m ttf.train --recipe convert-qwen --init-hf PATH/TO/Qwen3.5-0.8B-Base --out-dir runs/qwen
python -m ttf.train --recipe convert-27b --init-hf PATH/TO/Qwen3.8-27B --kd-cache KD_DIR --out-dir runs/qw27   # docs/CONVERT_27B.md
```

Add `--resume` to continue any run; any option on the command line overrides the recipe's. `python -m ttf.train
--help` lists every option with its default (the measured best). Commands from before 17 Oct 2026 (the flags of tag
`pre-consolidation-2026-10-17`) translate with `ttf.args.from_legacy`; [docs/OPTIONS_MAP.md](docs/OPTIONS_MAP.md)
lists the correspondence. The earlier methods (the stateless flip rule, low-rank momentum, look-ahead, TTF with a
short momentum) run from a checkout of that tag (`python -m bitnet.train` there; ttf/legacy, its copy, was removed from the core on 25 Oct).

## Main options

| option | what |
|---|---|
| `--mode ttf / bitnet / fp32` | TTF; BitNet b1.58 (latent weights, STE, AdamW); the full-precision model |
| `--recipe NAME` | defaults for a setting (`recipes/`: scratch-110m, convert-110m, convert-qwen, convert-27b, continue-27b) |
| `--ttf-rank R` | rank of each layer's accumulator (512) |
| `--ttf-theta`, `--ttf-tau`, `--ttf-step-p P`, `--ttf-leak annealed / fixed` | threshold, leak, step scale ρ^P (ρ = lr / peak lr), leak scaled with the step |
| `--ttf-update fused / after` | each layer's update inside its backward pass (one layer's gradient at a time) |
| `--ttf-accum micro / sum` | with `--accum M`: the micro-batch update, no dense gradient |
| `--dw factored / dense`, `--dw-sketch K` | the update from the gradient's factors, the dense weight gradient never formed |
| `--ttf-store fp32 / fp12s / int8 / ...` | compressed accumulator |
| `--ttf-offload host / async`, `--ttf-offload-keep F` | state in pinned host memory (all but the first share F of layers); async: copies overlapped with the backward |
| `--adapter-rank R`, `--adapter-init zero / svd` | rank-R float adapter beside the trits (16) |
| `--init-hf DIR`, `--convert gptqc / gptq / als / ...` | convert a pretrained model (Llama, Qwen3.5 / 3.8, layer by layer) |
| `--teacher DIR`, `--kd-weight W`, `--kd-cache DIR` | distillation from the full-precision model, or from cached top-k targets |
| `--ttf-start S` | trits frozen until step S (conversion) |
| `--tern-emb tied / untied / tied_wide:M` | ternary token embedding (and head), trained by TTF |
| `--moe E:K[:H]` | mixture of ternary experts, each with its own state |

## Layout

```
ttf/          the package
  train.py       training loop (python -m ttf.train); args.py: the options, recipes, from_legacy
  update.py      TTF: ts_layer (the per-layer update), the factored update, state codecs, host offload
  layers.py      the ternary layer (packed trits, scales, adapter, gradient capture / fused hooks)
  kernel.py      Triton kernels (int8 / bf16 GEMMs, input gradient, packing, firing, codecs, fused norms)
  schedule.py, data.py, eval.py   lr and TTF schedules, batches and KD-cache windows, validation loss
  hf_init.py     conversion of pretrained Llama models (ALS, GPTQ onto the ternary grid, centred GPTQ)
  qwen.py        Qwen3.5 / 3.8 (Gated DeltaNet + attention) with ternary linears; streaming conversion
  master.py      BitNet b1.58 (--mode bitnet); tern_emb.py: ternary embeddings; opt_lean.py: LeanAdam
  model.py, config.py   Llama-style transformer, presets (small = 110M, d1024_l24 = 340M, d2048_l24 = 1.3B, b27)
  results.py     the results database (results/runs.jsonl): runs, events, results, notes
recipes/       TOML defaults for the measured settings (scratch-110m, convert-110m, convert-qwen, convert-27b, continue-27b)
scripts/
  results.py     the database's command line (status, table, show, find, note, ingest, validate)
  infer/         ttf_run.py (our models), ttf_qwen.py + ttf_kernels.py (converted Qwen, 2-bit dp4a decode)
  eval/          final_eval.py, compare.py, zeroshot*.py
  quant/, analysis/, data/, plots/   conversion tools, diagnostics, data preparation, figures
  tests/         kernel checks and benchmarks, det_run.py (deterministic runs)
tests/         pytest (-m "not gpu" anywhere; the gpu tests on a CUDA machine); checks/: script checks
docs/          see below
```

## Docs

- `results/runs.jsonl`, `python scripts/results.py`: every run with its configuration, events and results (`table`,
  `show`, `find`); notes are kept outside the repository (`ttf/results.py`, `TTF_NOTES`).
- [docs/FORMULAS.md](docs/FORMULAS.md): every update rule used, with its flag.
- [docs/OPTIONS_MAP.md](docs/OPTIONS_MAP.md): the old flags and the current options.
- [docs/CONVERT_27B.md](docs/CONVERT_27B.md): converting a 27B model on one 24 GB GPU (data, teacher cache, training,
  packing, inference).
- [docs/research/](docs/research/): early results summary, related work, analysis outputs (rank spectra, gaps,
  zero-shot).

## License and patents

The code is licensed under the PolyForm Noncommercial License 1.0.0 ([LICENSE](LICENSE)): free for any noncommercial
purpose, including personal research, study and use by educational, charitable, public research and government
organisations; that licence also covers the patent claims you would infringe by using this software for those
purposes. The TTF training and inference methods are the subject of UK patent applications GB2623290.0 and
GB2624374.1 (patent pending). Commercial use of the software or the methods needs a separate licence: contact the
author. Contributions are accepted under the Contributor Licence Agreement in [CONTRIBUTING.md](CONTRIBUTING.md).
