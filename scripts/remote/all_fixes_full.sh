#!/usr/bin/env bash
# Everything that may help, one full 300M run from scratch: rank-256 momentum + cross-batch look-ahead x2,
# additive low-rank magnitude (rank 16), sign gate, vnorm 0.99, spend 2, refresh 16/10, per-head temperature.
# Snapshots every 1000 steps for the head analysis. Waits for magadd16_qk, then for the GPU to be free
# (mul60_heads may run first if it was started).
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
until grep -q "DONE magadd16_qk" checkpoints/remote_queue.log 2>/dev/null; do sleep 60; done
sleep 240
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 60; done
N=all_fixes_full
rm -rf checkpoints/$N; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 --snap_every 1000 \
  --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
  --lowrank_mag add:16 --lr_gate --lr_vnorm 0.99 --lr_spend 2 --lr_refresh 16 --lr_refresh_every 10 --qk_temp \
  --out_dir checkpoints/$N > checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log)"
