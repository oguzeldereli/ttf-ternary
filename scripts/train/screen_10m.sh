#!/usr/bin/env bash
# 10M-token screen (316 steps) of the current best flip config, same seed and batch
# order as every other run. Usage: scripts/train/screen_10m.sh NAME [extra train flags]
# Extra flags are appended, so they override (e.g. --rate_peak 0.04).
set -eu
cd "$(dirname "$0")/../.."
NAME=${1:?run name}; shift
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
mkdir -p "checkpoints/$NAME"
exec python -m ttf.train --preset small --mode kernel \
  --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 \
  --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 \
  --lookahead 1 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 316 --probe 0-40:5,40-320:20 \
  --eval_iters 30 --eval_interval 100000 --save_secs 100000 \
  --out_dir "checkpoints/$NAME" "$@" > "checkpoints/$NAME/train.log" 2>&1
