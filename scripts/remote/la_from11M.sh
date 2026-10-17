#!/usr/bin/env bash
# Momentum switched off at 11M (step 340, where M is already noise): continue the momentum +
# look-ahead run with look-ahead x2 alone (gradient proposals), same rate schedule as the momentum run, same batches, to the end of the
# schedule (300M). Is the later part of the momentum run pure look-ahead?
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
IN=$HOME/ternary-sync/inbox/ckpt_340.pt
[ -f $S/ckpt_340.pt ] || { cp $IN $S/ckpt_340.pt && rm -f $IN; }
N=r4090_la_from11M
mkdir -p checkpoints/$N
[ -e checkpoints/$N/ckpt.pt ] || ln -s $S/ckpt_340.pt checkpoints/$N/ckpt.pt
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips \
  --track_reversals --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 9155 --eval_iters 30 \
  --eval_interval 250 --save_secs 3600 --lookahead 2 --lookahead_xbatch --ckpt_skip 2 --resume \
  --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log)"
