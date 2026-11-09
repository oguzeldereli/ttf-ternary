# ttf-ternary

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23394608.svg)](https://doi.org/10.5281/zenodo.23394608)
[![tests](https://github.com/oguzeldereli/ttf-ternary/actions/workflows/tests.yml/badge.svg)](https://github.com/oguzeldereli/ttf-ternary/actions/workflows/tests.yml)

Train ternary ({-1, 0, +1}) language models **without full-precision master weights**. The weights stay packed at 1.6
bits (5 trits per byte) for the whole of training. The optimizer state is low-rank, about one byte per weight in place
of the 12-16 bytes of master weights plus Adam. A 23B-parameter model trains on one 16 GB GPU, and a converted
Qwen3.8-27B trains on one 24 GB card.

The method is **two-timescale threshold flips (TTF)**. The Adam-normalised gradient (with a factored second moment)
feeds a leaky accumulator kept at low rank. A weight moves one level when its accumulated evidence crosses a threshold,
and the threshold is then spent from the accumulator. The trits are the only per-weight state.

Paper: [*Training Ternary Language Models Without Master Weights*](paper/ttf_paper1_training.pdf) (in this repository;
LaTeX source in [paper/](paper/)). It has every result below with its run, and an appendix of every mechanism tried.

**The TTF family**

| repository | what it does |
|---|---|
| **ttf-ternary** (this one) | the trainer: TTF from scratch or from a ternary checkpoint, Llama-style models and Qwen3.5 / 3.8 (dense and Flash-Next MoE), BitNet b1.58 and full-precision baselines, evaluation |
| [ttf-conversion](https://github.com/oguzeldereli/ttf-conversion) | converts a pretrained Hugging Face model to a ternary checkpoint this trainer continues from |
| [ttf-home](https://github.com/oguzeldereli/ttf-home) | one command line for a home computer: prepare data, train, convert, fine-tune, export to GGUF, chat / serve with llama.cpp |
| [spanqr](https://github.com/oguzeldereli/spanqr) (optional) | a faster robust orthonormal basis for the accumulator, used automatically when installed |

**Released models** (Hugging Face; GGUF files for ttf-home's llama.cpp):

| model | file | GSM8K (first 200) | HumanEval | runs on |
|---|---|---|---|---|
| [ttf-Qwen3.8-27B-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-27B-GGUF) | 10.2 GB | 0.895 (full precision 0.895) | 0.841 (0.933) | one 12 GB GPU |
| [ttf-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-Flash-Next-GGUF) | 40.4 GB | 0.905 (FP8 0.905) | 0.848 (0.927) | a 24 GB GPU + 64 GB RAM (experts on the CPU) |

## Requirements

- Linux with an NVIDIA GPU (CUDA). The kernels are Triton and use the int8 tensor cores; measured on RTX 3090 Ti,
  4070 Ti Super, 4080 Laptop and RTX PRO 6000. A 16 GB card trains up to ~1B parameters directly and much larger
  models with the state offloaded to host memory.
- Python 3.11+, PyTorch 2.13+ (with its Triton).
- macOS and Windows are not supported for training. The CPU runs the test suite only. To *run* the trained models on
  any machine, use [ttf-home](https://github.com/oguzeldereli/ttf-home) (llama.cpp: CUDA, Metal, Vulkan or CPU).

## Install

```bash
pip install "ttf-ternary[data] @ git+https://github.com/oguzeldereli/ttf-ternary"
# or, to edit the code and use the recipes and scripts:
git clone https://github.com/oguzeldereli/ttf-ternary && cd ttf-ternary && pip install -e ".[data,test]"
```

Extras: `[qwen]` for Qwen3.5 / 3.8 (transformers 5.19, flash-linear-attention), `[data]` for the data scripts,
`[test]` for pytest. Optional, faster at ranks of 256 and up: clone
[spanqr](https://github.com/oguzeldereli/spanqr) and put it on `PYTHONPATH`; `--ttf-orth auto` (the default) uses it
when it imports.

Check the install: `pytest -q -m "not gpu"` (CPU, a few seconds) and `pytest -q` on the GPU machine.

## Quickstart: a few minutes on one GPU

A 15M model from scratch on 20M tokens of Wikipedia, TTF against BitNet b1.58 (master weights + straight-through
estimator) on the same data and schedule ([recipes/demo-15m.toml](recipes/demo-15m.toml)):

```bash
python3 scripts/data/prep_wiki.py 2e7 --tokenizer llama --val_tokens 1000000 --out_prefix data/demo   # ~5 min download
python -m ttf.train --recipe demo-15m --out-dir runs/demo-ttf
python -m ttf.train --recipe demo-15m --mode bitnet --out-dir runs/demo-bitnet
python -m ttf.train --recipe demo-15m --mode fp32 --out-dir runs/demo-fp32          # the full-precision reference
```

DEMO_TABLE

Every run writes `train.log`, `metrics.jsonl` and `ckpt.pt` to its `--out-dir`; `--resume` continues it.

## The paper's 110M run

```bash
python3 scripts/data/prep_wiki.py 4e8 --tokenizer llama --out_prefix data/wiki32k    # 400M tokens of Wikipedia
python -m ttf.train --recipe scratch-110m --out-dir runs/ttf                          # ~3.5 h on a 4070 Ti Super
python -m ttf.train --recipe scratch-110m --mode bitnet --out-dir runs/bitnet
python scripts/eval/final_eval.py runs/ttf runs/bitnet                                # all 974 validation windows
python scripts/eval/compare.py runs/ttf runs/bitnet --set fw                          # paired, on FineWeb-Edu
```

## Train a converted model

Convert a pretrained model with [ttf-conversion](https://github.com/oguzeldereli/ttf-conversion), then train the
ternary checkpoint here with distillation from the original. Qwen3.5-0.8B on one 16 GB GPU:

```bash
pip install -e ".[qwen]" "ttf-conversion @ git+https://github.com/oguzeldereli/ttf-conversion"
hf download Qwen/Qwen3.5-0.8B-Base --local-dir models/Qwen3.5-0.8B-Base
M=models/Qwen3.5-0.8B-Base
# the Wikipedia tokens of the 110M run, re-tokenized for Qwen (uint32)
python scripts/quant/retok.py data/wiki32k_train.bin data/wikiqw_train.bin --target $M --max_tokens 6e7
python scripts/quant/retok.py data/wiki32k_val.bin data/wikiqw_val.bin --target $M
# convert: norm gains folded + a Hadamard rotation of the residual stream, ternary GPTQ, ternary low-rank adapters
python -m ttfconv.convert --model $M --data data/wikiqw_train.bin --out runs/conv08 \
    --ext ttfconv.fold --ext "ttfconv.adapters:rank=0.2,alt=1"
# 1000 steps of TTF recovery (32k tokens per step; ttf-conversion's recipe)
python -m ttf.train --recipe PATH/TO/ttf-conversion/recipes/qwen-0.8b.toml --arch-hf $M --init-ckpt runs/conv08/ckpt.pt \
    --teacher $M --data data/wikiqw_train.bin --val data/wikiqw_val.bin --batch-size 2 --accum 8 \
    --ext ttfconv.fold --ext "ttfconv.adapters:rank=0.2,alt=1" --out-dir runs/q08
```

Qwen3.8-27B: ttf-conversion's [docs/CONVERT_27B.md](https://github.com/oguzeldereli/ttf-conversion/blob/main/docs/CONVERT_27B.md)
and `--recipe continue-27b` here (2 x 8 micro-batches of 2048 tokens on one 24 GB GPU, a cached teacher with
`--kd-cache`). Qwen3.8-Flash-Next (512 experts per layer) has its own layer-major trainer: [docs/Q38.md](docs/Q38.md).

## Run a trained model

- **GGUF / llama.cpp:** `ttf export` in [ttf-home](https://github.com/oguzeldereli/ttf-home) writes an ordinary GGUF
  file. The export is exact (llama.cpp's next-token loss matches the trainer's).
- **In Python:** `python scripts/infer/ttf_run.py run runs/demo-ttf/ckpt.pt -p "The history of"` (models trained
  here; `ppl` scores a token file, `export` writes a small safetensors file) and `scripts/infer/ttf_qwen.py` (converted
  Qwen, trits never expanded: 22.7 tokens/s for the 27B on a 12 GB laptop GPU).

## Results

All numbers are validation loss (nats per token) on fixed windows: all 974 non-overlapping 2048-token windows of the
Wikipedia validation set, and 2000 windows of FineWeb-Edu, which was never used for any choice
(`scripts/eval/final_eval.py`). "Master + extras" is BitNet b1.58 with exactly the auxiliary float parameters TTF has
(row / column scales, rank-16 float adapter, attention temperature): the fair reference.

**110M, 300M tokens of Wikipedia (Llama-2 32k tokenizer)**

| method | Wikipedia | FineWeb-Edu |
|---|---|---|
| **TTF, rank 512 (the paper's recipe)** | **2.6728** | **3.6226** |
| TTF, the default θ schedule (seeds 1337 / 2) | 2.6741 / 2.6695 | |
| TTF, full rank | 2.6480 | 3.5976 |
| TTF, rank 512, no float extras | 2.6907 | 3.6435 |
| master + extras | 2.7166 | 3.6830 |
| master (BitNet b1.58) | 2.736 | 3.707 |
| low-rank Adam STE, rank 512 | 2.8867 | 3.8678 |

At 600M tokens of FineWeb-Edu TTF leads master + extras by 0.056 (3.0423 vs 3.0982).

**Scale (TTF minus master + extras at its best learning rate)**

| size | TTF state (numbers per weight) | Wikipedia | FineWeb-Edu |
|---|---|---|---|
| 340M | rank 512 (0.79) | −0.032 | −0.042 |
| 340M | full rank 1024 (1.57) | −0.071 | −0.087 |
| 1.3B | rank 1024 (0.79), two seeds | −0.010 | −0.027 |
| 1.3B | full rank 2048 (1.57) | −0.051 | −0.069 |
| 2.9B (trained on FineWeb-Edu) | rank 1536 (0.79) | +0.002 | −0.004 |

TTF's gap to master is a smooth function of the size of its state, and closes at 0.57-0.70 numbers of state per
weight at every size: about one byte per weight at the 12 bits the accumulator tolerates, against 12 bytes for master.

**Converted pretrained models** (the conversion is ttf-conversion's; the recovery training is TTF here)

| model, recovery tokens | full precision | converted, before training | after |
|---|---|---|---|
| TinyLlama-1.1B, 50M | 1.875 | 8.01 | **2.2320**; with distillation 2.1878 |
| Qwen3.5-0.8B-Base, 50M | 2.556 | 4.96 | 2.8503 |
| Qwen3.8-27B, 72M (the released model) | — | — | GSM8K 0.895 (full precision 0.895), HumanEval 0.841 (0.933), IFEval 0.735 (0.787) |

## Memory and speed

- **23.0B parameters on one 16 GB GPU** (22.85B ternary; 11.1-11.6 GiB at ranks 256-1024) with the TTF state in host
  memory and copies overlapped with the backward (`--ttf-offload async`).
- **Qwen3.8-27B on one 24 GB GPU:** 32k tokens per step (2 x 8 x 2048), peak 19.7 GB (`--recipe continue-27b`).
- **Micro-batches without a dense gradient** (`--accum M`): each micro-batch's gradient enters the state and is
  dropped; 4 x 4 micro-batches match one pass of batch 16 (2.6733 vs 2.6728) at 3.1 instead of 7.4 GiB.
- **12-bit state** (`--ttf-store fp12s`) matches 32-bit; at equal memory a higher rank in 12 bits beats a lower rank in
  32 (rank 340 at 12 bits 2.7086 vs rank 128 at 32 bits 2.8410).
- **Checkpointing is chosen from measured memory** (`--ckpt-skip auto`, the default): blocks keep their activations
  while the GPU has room, then the int8 trit cache.
- **Use the largest micro-batch that fits:** for the same tokens per step, 2 x 8 is 25% faster than 1 x 16 on a
  0.8B conversion (11.4 vs 15.1 s per step on a 4070 Ti Super).
- **The loss head** of large vocabularies (Qwen's 248k) runs in one Triton kernel: 3.3x faster and 2.4 GiB less
  transient memory per 512-token chunk than the PyTorch ops.

## Main options

`python -m ttf.train --help` lists every option with its default, which is the measured best. A recipe
(`--recipe NAME`, [recipes/](recipes/)) sets defaults for a setting; any option on the command line overrides it.

| option | what |
|---|---|
| `--mode ttf / bitnet / fp32` | TTF; BitNet b1.58 (latent weights, STE, AdamW); the full-precision model |
| `--recipe NAME` | demo-15m, scratch-110m, continue-27b |
| `--preset NAME` | model shape without `--arch-hf` (s15, small = 110M, d1024_l24 = 340M, d2048_l24 = 1.3B, b3, b27, ...) |
| `--ttf-rank R` | rank of each layer's accumulator (512) |
| `--ttf-theta`, `--ttf-tau` | base threshold (θ alone sets the step), leak 1 − 1/τ |
| `--ttf-theta-start`, `--ttf-theta-end` | θ schedule: start at step 0, base at the warmup's end, end at the last step, cosines between them in the step base/θ (default 4θ → θ → 5θ; start = end = base for a constant θ) |
| `--accum M`, `--batch-size B` | M micro-batches of B sequences per step |
| `--dw-sketch K`, `--dw-sketch-warm F` | the update from the gradient's factors (no dense weight gradient), sketch size K |
| `--ttf-store fp32 / bf16 / fp12s` | accumulator format |
| `--ttf-offload async` | TTF state in pinned host memory, copies overlapped with the backward |
| `--ttf-orth auto / qr / spanqr` | the accumulator basis' orthonormalisation |
| `--adapter-rank R` | rank-R float adapter beside the trits (16) |
| `--arch-hf DIR`, `--init-ckpt CKPT` | a Hugging Face architecture (Llama, Qwen3.5 / 3.8) and a ternary checkpoint of it |
| `--teacher DIR`, `--kd-weight W`, `--kd-cache DIR` | distillation from the full-precision model, or from cached top-k targets |
| `--ext MODULE[:SPEC]` | a structure plug-in (ttf-conversion's fold, ternary adapters, ...) |
| `--tern-emb`, `--moe E:K[:H]` | ternary token embedding; a mixture of ternary experts |

## Layout

```
ttf/            the package
  train.py        training loop (python -m ttf.train / ttf-train); args.py: options, recipes
  update.py       TTF: the per-layer update, the factored update, state codecs, host offload
  layers.py       the ternary layer (packed trits, scales, adapter, fused forward / backward)
  kernel.py       Triton kernels (int8 GEMMs, input gradient, packing, firing, codecs, fused norms)
  headloss.py     the fused loss head (CE + distillation and its gradient in one pass)
  memplan.py      --ckpt-skip auto
  qwen.py         Qwen3.5 / 3.8 (Gated DeltaNet + attention) with ternary linears, pipeline stages
  q38/            Qwen3.8-Flash-Next (MoE) in TTF: batched expert TTF, grouped GEMMs, layer-major trainer
  model.py, config.py, master.py, data.py, eval.py, schedule.py, ...
recipes/        demo-15m, scratch-110m, continue-27b
scripts/        data/ (Wikipedia, FineWeb-Edu), eval/, infer/, prof/ (step time and memory profilers), analysis/
tests/          pytest (-m "not gpu" on any machine; the rest on a CUDA GPU)
examples/vit/   a CIFAR-100 ViT: TTF vs master weights vs full precision
docs/           FORMULAS.md (the update in full), Q38.md (Flash-Next), OPTIONS_MAP.md (old flag names)
paper/          the paper (PDF, LaTeX, figures)
results/        runs.jsonl: every run behind the paper, with its configuration (python scripts/results.py table)
```

## Citation

```bibtex
@software{guzeldereli_ttf_ternary,
  author = {G{\"u}zeldereli, {\"O}zg{\"u}r},
  title  = {ttf-ternary: training ternary language models without master weights},
  year   = {2026},
  doi    = {10.5281/zenodo.23394608},
  url    = {https://github.com/oguzeldereli/ttf-ternary}
}
```

## License

PolyForm Noncommercial 1.0.0 ([LICENSE](LICENSE)): free for any noncommercial purpose, including personal use,
research and study, and use by educational, charitable, public research and government organisations. Commercial use
needs a separate licence: contact the author. The released model weights keep their base models' licence (Apache-2.0
for Qwen). Contributions are welcome under [CONTRIBUTING.md](CONTRIBUTING.md).
