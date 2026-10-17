#!/usr/bin/env bash
# Resume rank-256 momentum without look-ahead (lowrank256_nola_full, copied to lowrank256_nola_laptop so the 4090 sync
# cannot overwrite it) from its 30M checkpoint (step 904) on the
# laptop, full 300M schedule, snapshots every 500 steps for the induction check (exact vs shuffled repeat).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
N=lowrank256_nola_laptop
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 500 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --resume --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log)"
