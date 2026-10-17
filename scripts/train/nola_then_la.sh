#!/usr/bin/env bash
# Branch of nola_lab (rank-256 momentum without look-ahead) at step 4000 = 131M tokens, after its induction heads
# formed: switch cross-batch look-ahead x2 on and continue to 164M (step 5000), snapshots every 250 steps.
# Question: is look-ahead an offset (loss gain) that keeps the formed circuit, or does it undo it?
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
N=nola_then_la
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 250 \
  --stop_after 5001 --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 --resume \
  --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
