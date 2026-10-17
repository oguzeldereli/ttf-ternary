#!/usr/bin/env bash
# 4090, priority, alone on the GPU: stop the nola_b48 filler (checkpoint saved), then accumulate-then-flip
# (--accum_flip 33, loss-chosen subset of the standout entries): branch of nola_lab at 131M -> 300M, then from
# scratch -> 300M; then resume nola_b48.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
pkill -u "$USER" -f "remote/night_4090.sh"; pkill -u "$USER" -f "remote/nola_b48_now.sh"; pkill -u "$USER" -f "remote/flip_selector_now.sh"
pkill -TERM -u "$USER" -f "out_dir checkpoints/nola_b48$"
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 5; done
run() {   # NAME START [flags]
  local N=$1 START=$2; shift 2
  mkdir -p checkpoints/$N
  if [ "$START" = branch ] && [ ! -f checkpoints/$N/ckpt.pt ]; then
    cp $S/bench_nola/ckpt_4000.pt checkpoints/$N/ckpt.pt
    cp $HOME/ternary-sync/inbox/bench_nola/metrics_4000.jsonl checkpoints/$N/metrics.jsonl
    echo "[branch of nola_lab at step 4000 (131M), no look-ahead: $*]" > checkpoints/$N/train.log
  fi
  python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
    --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
    --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
    --lookahead 0 --lowrank 256 --ckpt_skip 2 "$@" $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) \
    --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
  echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
}
run accum33_b131 branch --accum_flip 33 --accum_z 3
run accum33_s0 scratch --accum_flip 33 --accum_z 3
bash scripts/remote/nola_b48_now.sh
