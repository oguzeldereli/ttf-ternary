#!/usr/bin/env bash
# Full 300M-token run (9155 steps, ~10 h) of the current best flip config:
# look-ahead + flip-rate ramp 0->0.02 over 30 steps then cosine to 0 + master's float-tail
# LR schedule (1.5e-3 -> 1.5e-4) + fp32 float tail. Resumable: rerun with --resume.
# Usage: scripts/train/full_run.sh NAME [extra train flags]
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
  --lr 1.5e-3 --min_lr 1.5e-4 --probe 0-40:5,40-9155:250 --snap_every 1000 \
  --eval_interval 1000 --eval_iters 30 \
  --out_dir "checkpoints/$NAME" "$@" >> "checkpoints/$NAME/train.log" 2>&1
