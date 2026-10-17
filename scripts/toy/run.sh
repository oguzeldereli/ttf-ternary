#!/usr/bin/env bash
# One toy2 run on the synthetic induction data. Usage: scripts/toy/run.sh NAME [extra train flags]
# Kernel mode by default; pass --mode master --master_dtype fp32 for the master control.
set -eu
cd "$(dirname "$0")/../.."
NAME=${1:?run name}; shift
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
D=checkpoints/toy/$NAME
rm -rf "$D"; mkdir -p "$D"
python -m ttf.train --preset toy2 --mode kernel --data data/toy_ind${TOY_TAG:-}_train.bin --val data/toy_ind${TOY_TAG:-}_val.bin \
  --seq_len 256 --batch_size 32 --grad_accum 1 --steps ${STEPS:-3000} --warmup 100 \
  --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 \
  --int8 --dw_mode dense --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 \
  --eval_interval 250 --eval_iters 20 --save_secs 100000 --snap_every 100 \
  --out_dir "$D" "$@" > "$D/train.log" 2>&1
