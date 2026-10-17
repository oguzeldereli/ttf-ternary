# Converting a 27B model on one 24 GB GPU

The steps that produced the ternary Qwen3.8-27B (`qw27_rel`, 16 Oct 2026: 2.2439 on 8 Wikipedia windows packed, full
precision 1.9034), with today's options. Everything runs on one RTX 3090 Ti / 4090 (24 GB) plus host memory; the
conversion and training never hold the full-precision model on the GPU.

| step | tool | time (3090 Ti) | needs |
|---|---|---|---|
| 1. model and tokenized validation text | `huggingface_hub`, `scripts/quant/retok.py` | minutes | 54 GB disk |
| 2. training mixture (optional: or any uint32 token stream) | `scripts/quant/build_mix3.sh` (`sft_mix.py`) | hours (CPU) | network |
| 3. teacher cache: the fp model's top-32 log-probs on the training windows | `scripts/quant/teacher_topk.py` | ~80 s per 48 windows | 64 GB host RAM |
| 4. conversion + training | `python -m ttf.train --recipe convert-27b` | 69 s per 32k-token step | |
| 5. packing for inference | `scripts/infer/pack_ttf.py` | minutes | |
| 6. evaluation, inference | `scripts/eval/qwen_zeroshot.py`, `scripts/infer/ttf_qwen.py` | | 12 GB GPU for inference |

Python packages beyond `requirements.txt`: `transformers==5.19.0 flash-linear-attention==0.5.2` (Qwen3.5 / 3.8's
Gated DeltaNet layers), `pyarrow` (step 2).

## 1. Model and validation text

```bash
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B', local_dir='models/qw27',
  allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])"
python scripts/data/prep_wiki.py 4e8 --tokenizer llama --out_prefix data/wiki32k          # once, Llama tokens
python scripts/quant/retok.py data/wiki32k_val.bin data/wikiq8_val.bin --target models/qw27   # Qwen tokens (uint32)
python scripts/quant/retok.py data/wiki32k_train.bin data/wikiq8_train.bin --target models/qw27
```

## 2. Training mixture

`MODEL=models/qw27 OUT=data/m3 WIKI=data/wikiq8_train.bin bash scripts/quant/build_mix3.sh` builds the ~800M-token
post-training mixture (Nemotron post-training math / code / stem / chat / tool, Tulu-3, OpenThoughts3,
OpenCodeReasoning, FineWeb-Edu, Wikipedia; rendered with the model's own chat template, decontaminated against the
evaluation items by word 13-grams: run `python scripts/eval/zeroshot_tok.py` first for those). `build_mix4s.sh` adds a
code- and reasoning-heavy 320M-token supplement. The release itself trained on an earlier 72M-token mixture of the
same kind (`sft_mix.py mix`, documented in its docstring).

## 3. Teacher cache

```bash
python scripts/quant/teacher_topk.py models/qw27 data/m3/mix3_train.bin kd3 --sequential --windows 0
```

The fp model is held in pinned host memory (bf16, ~54 GB) and streamed through the GPU one decoder layer at a time, over
48 windows per shard. Several GPUs or machines fill one cache: `--claim_dir DIR` on a shared filesystem (each machine
claims the next free shard), or `--part R/N` (machine R of N); then copy all shards into one directory and, once it
is complete, `touch DIR/done`. The training run reads shard windows in order from `--kd-cache-start`; a run that
outpaces the cache waits for the next shard, and after `done` it wraps around the shards it has.
`teacher_gguf.py` does the same from a GGUF quantized teacher through llama.cpp (`llama_topk.cpp`).

## 4. Conversion and training

```bash
python -m ttf.train --recipe convert-27b --init-hf models/qw27 --kd-cache kd3 --out-dir runs/qw27
```

`recipes/convert-27b.toml`: streaming GPTQ conversion (layer by layer from the safetensors, 64 calibration windows),
trits frozen for 400 steps while the floats (scales, rank-16 adapters, norms; LeanAdam) calibrate, then TTF at theta
64 with rank-256 accumulators stored in 12 bits, the update from the gradient's factors (no dense weight gradient) and
the firing fused; loss (1 - 0.5) CE + 0.5 KD to the cached top-k; 2 x 8 micro-batches of 2048 tokens; PyTorch capped
at 22.4 GiB. Resume with `--resume` after any stop (SIGTERM saves). A 32 GB card can take `--batch-size 4 --accum 4`.
Continuing a finished run: `recipes/continue-27b.toml` (note: the 17 Oct continuations of qw27_rel at theta 16 and 64
raised validation; not a settled recipe).

## 5-6. Packing, evaluation, inference

```bash
python scripts/infer/pack_ttf.py runs/qw27/ckpt.pt qw27_e4.pt --emb_bits 4 --head_bits 4      # 6.6 GB
python scripts/eval/qwen_zeroshot.py models/qw27 --ttf qw27_e4.pt --limit 2000
python scripts/infer/ttf_qwen.py models/qw27 qw27_e4.pt --chat -i                              # 2-bit decode kernels
```
