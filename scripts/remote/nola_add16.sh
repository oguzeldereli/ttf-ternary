#!/usr/bin/env bash
# Rank-256 momentum WITHOUT look-ahead + additive adapter rank 16 (no weight decay, no temperature), from scratch,
# full 300M schedule, snapshots every 500 steps. The no-look-ahead run (nola_lab) forms induction at ~98M; this
# asks whether the adapter's loss gain carries over without look-ahead. Waits for the 131M extension.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
until grep -q "DONE magadd16_wd_qk_131M" checkpoints/remote_queue.log 2>/dev/null; do sleep 30; done
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 20; done
N=nola_add16
rm -rf checkpoints/$N; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 500 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --lowrank_mag add:16 \
  --out_dir checkpoints/$N > checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log)"
