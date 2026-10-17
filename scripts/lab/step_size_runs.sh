#!/usr/bin/env bash
# Lab PC, side by side, branches of nola_lab at 131M -> 300M, no look-ahead:
#   small_step_b131  plain rank-256 momentum at 1/4 of the flip rate (--rate_peak 0.005)
#   accum33_b131     accumulate 33 steps without flips, then flip a loss-chosen random subset of the entries whose
#                    accumulated push stands out (|M| > 3 rms), restart the momentum
set -u
L=$HOME/ternary-LLM/scripts/lab
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
pgrep -u "$USER" -f "scripts/lab/sync_out.sh" >/dev/null || setsid nohup bash $L/sync_out.sh >/dev/null 2>&1 &
run() {   # NAME [flags]
  local N=$1; shift
  mkdir -p checkpoints/$N
  if [ ! -f checkpoints/$N/ckpt.pt ]; then
    cp $S/bench_nola/ckpt_4000.pt checkpoints/$N/ckpt.pt
    cp $HOME/ternary-sync/inbox/bench_nola/metrics_4000.jsonl checkpoints/$N/metrics.jsonl
    echo "[branch of nola_lab at step 4000 (131M), no look-ahead: $*]" > checkpoints/$N/train.log
  fi
  python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
    --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
    --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
    --lookahead 0 --lowrank 256 --ckpt_skip 2 "$@" --resume --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
  echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> $HOME/ternary-sync/lab_queue.log
}
run small_step_b131 --rate_peak 0.005 &
run accum33_b131 --accum_flip 33 --accum_z 3 &
wait
