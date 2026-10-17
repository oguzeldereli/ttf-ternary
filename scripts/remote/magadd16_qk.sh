#!/usr/bin/env bash
# Additive low-rank magnitude (rank 16) + per-head attention temperature, momentum + look-ahead x2, from scratch,
# full schedule stopped at 200M tokens (step 6104): do induction heads form (master: ~100-160M)?
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
N=magadd16_qk
rm -rf checkpoints/$N; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 --snap_every 1000 \
  --stop_after 6105 --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
  --lowrank_mag add:16 --qk_temp --out_dir checkpoints/$N > checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log)"
