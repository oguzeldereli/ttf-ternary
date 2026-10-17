#!/usr/bin/env bash
# Momentum + look-ahead on a schedule, from scratch, full 300M: look-ahead ON to 30M (step 915, past the
# no-look-ahead dead zone at 2-4M), OFF to 131M (step 4000, so induction can form), ON again after (the
# nola_then_la branch kept its induction and recovered the loss). Snapshots every 500 steps. Waits for the GPU.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 20; done
N=la_sched
mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 500 \
  --lookahead 2 --lookahead_xbatch --lookahead_off 915:4000 --lowrank 256 --ckpt_skip 2 \
  $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
