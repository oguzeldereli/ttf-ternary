#!/usr/bin/env bash
# 4090: same as runner_lab.sh with the 4090 recipe; waits for any running training first (not for queue_4090_c.sh by name: the tmux server's command line keeps that name).
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM; L=$REPO/scripts/night/4090.list
source $S/env.sh; cd $REPO
while pgrep -u "$USER" -f "[p]ython -m ttf.train" >/dev/null; do sleep 30; done
while line=$(grep -m1 -v '^#' "$L" 2>/dev/null) && [ -n "$line" ]; do
  grep -v -x -F "$line" "$L" > "$L.tmp"; mv "$L.tmp" "$L"
  set -- $line; N=$1; shift
  mkdir -p checkpoints/$N
  python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
    --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
    --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
    --lowrank 256 --ckpt_skip 2 --lookahead 0 "$@" $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) \
    --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
  echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
done
