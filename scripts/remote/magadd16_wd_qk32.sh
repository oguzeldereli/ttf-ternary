#!/usr/bin/env bash
# Additive rank 16 with weight decay 0.1 on the adapter (the toy's induction fix) + per-head temperature,
# momentum + look-ahead x2, from scratch, full 300M schedule stopped at 32M tokens (step 976), snapshots
# every 250 steps for the induction check. Resumable later (ckpt.pt).
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 20; done
N=magadd16_wd_qk
rm -rf checkpoints/$N; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 --snap_every 250 \
  --stop_after 977 --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
  --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --out_dir checkpoints/$N > checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log)"
