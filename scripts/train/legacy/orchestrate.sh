#!/usr/bin/env bash
# Run queue used for the flip-rule study. Each run resumes from its own
# checkpoint if interrupted; see RUNS.md for the results.
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
LOG=checkpoints/orchestrate.log
say() { echo "[$(date +%m-%d\ %H:%M)] $*" | tee -a $LOG; }

ACC1_CFG="--seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305"
KERNEL="--preset small --mode kernel --data data/wiki32k_train.bin \
--val data/wiki32k_val.bin --lr 3e-4 --min_lr 3e-5 --eval_interval 1000 --eval_iters 30 \
--track_flips --int8 --dw_mode dense --save_secs 600"
BASE="--preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
--val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 \
--warmup 305 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000 --eval_iters 30 --save_secs 600"

final_ppl() { grep -oE "ppl [0-9.]+" "$1" 2>/dev/null | tail -1 | awk '{print $2}'; }

run() {
  local name=$1; shift
  local out=checkpoints/$name
  if [ -f "$out/done" ]; then say "$name already done ($(final_ppl $out.log))"; return 0; fi
  mkdir -p "$out"
  say "START $name"
  # retry once on crash: --resume picks up from the last checkpoint
  python -m ttf.train "$@" --out_dir "$out" >> "$out.log" 2>&1 \
    || { say "RETRY $name after failure"; python -m ttf.train "$@" --resume --out_dir "$out" >> "$out.log" 2>&1; }
  if [ $? -eq 0 ]; then
    touch "$out/done"; say "DONE $name -> ppl $(final_ppl $out.log)"
    python plot_run.py "$out" --title "$name" >> $LOG 2>&1 || true
  else
    say "FAILED $name (see $out.log)"; tail -5 "$out.log" | tee -a $LOG
  fi
}

A1=135.84

# armB is running (resumed by queue 12). Wait for it, finish its bookkeeping.
until [ "$(grep -c '^done$' checkpoints/armB_absscale.log 2>/dev/null)" -ge 1 ]; do sleep 60; done
touch checkpoints/armB_absscale/done
say "DONE armB_absscale -> ppl $(final_ppl checkpoints/armB_absscale.log)"
python plot_run.py checkpoints/armB_absscale --title armB_absscale >> $LOG 2>&1 || true

# The winning recipe (cosine rate 2e-2 -> 0) at DOUBLE the token budget: 600M.
# Cosine spans the full 18310 steps, so the anneal still lands on 0 at the end.
run armA_cos_600M $KERNEL --seq_len 2048 --batch_size 16 --grad_accum 1 \
  --steps 18310 --warmup 610 --rate_schedule cosine --rate 0.02

say "QUEUE 13 COMPLETE (acc4 extension dropped per request)"
