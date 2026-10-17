#!/usr/bin/env bash
# Control: gradient proposals + look-ahead x2 at rate 0.0072 (flip count matched to branch A),
# 60 steps from ckpt_340, same batches; then resume the main run.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 \
  --rate_schedule cosine --rate 0.0 --rate_peak 0.0072 --rate_warmup 30 --g_ref 3.0 \
  --int8 --dw_mode dense --track_flips --track_reversals --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 \
  --stop_after 400 --eval_iters 30 --eval_interval 100000 --save_secs 100000 --ckpt_skip 2 \
  --lookahead 2 --lookahead_xbatch --resume --out_dir checkpoints/branch340_C > checkpoints/branch340_C/train.log 2>&1
exec scripts/train/continue_lowrank_xb2_300M.sh
