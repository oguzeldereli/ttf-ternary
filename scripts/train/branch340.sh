#!/usr/bin/env bash
# From ckpt_340 (11M tokens, where the momentum run is level with master): 60 steps of
# A = momentum proposals + cross-batch look-ahead x2, B = gradient proposals + the same filter.
# Same batches (resume fast-forwards the data streams); reversals and net displacement logged.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
base="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305
  --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
  --int8 --dw_mode dense --track_flips --track_reversals --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4
  --stop_after 400 --eval_iters 30 --eval_interval 100000 --save_secs 100000 --ckpt_skip 2
  --lookahead 2 --lookahead_xbatch --resume"
python -m ttf.train $base --out_dir checkpoints/branch340_A --lowrank 256 --lr_diag > checkpoints/branch340_A/train.log 2>&1
python -m ttf.train $base --out_dir checkpoints/branch340_B > checkpoints/branch340_B/train.log 2>&1
exec scripts/train/continue_lowrank_xb2_300M.sh
