#!/usr/bin/env bash
# After the no-look-ahead resume: momentum + look-ahead x2 WITH the sign gate, from scratch, full 300M schedule
# (the long-horizon test of the gate; compare lm_lowrank256_xb2_100M, val 3.127).
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
until grep -q "DONE lowrank256_nola_full" checkpoints/remote_queue.log 2>/dev/null; do sleep 60; done
N=gate_full
rm -rf checkpoints/$N; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 \
  --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 --lr_gate --out_dir checkpoints/$N \
  > checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log)"
