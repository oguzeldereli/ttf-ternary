#!/usr/bin/env bash
# What happens at the ~12M-token bend? From scratch to 30M tokens with in-training probes every 25 steps
# (copy_gain = induction, loss_ctxK = context use): full precision, master, ours; plus full precision with a
# 30- and a 900-step LR warmup (does the bend follow the end of warmup?). Then the remaining fix trials.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--preset small --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
  --grad_accum 1 --steps 9155 --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 916 --eval_interval 250 --eval_iters 30
  --save_secs 100000 --probe 0-920:25"
one() {
  local n=$1; shift
  rm -rf checkpoints/$n; mkdir -p checkpoints/$n
  python -m ttf.train $COMMON "$@" --out_dir checkpoints/$n > checkpoints/$n/train.log 2>&1
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
one bend_fp32 --mode fp32 --warmup 305
one bend_master --mode master --master_dtype fp32 --warmup 305
one bend_fp32_w30 --mode fp32 --warmup 30
one bend_fp32_w900 --mode fp32 --warmup 900
one bend_ours --mode kernel --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 \
  --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2
# then the remaining overnight fix trials
run() {
  local n=$1 steps=$2; shift 2
  scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
    --stop_after $steps --eval_interval 200 --save_secs 3600 "$@"
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
run s20_gate_vnorm99 611 --lr_gate --lr_vnorm 0.99
run s20_base_seed2 611 --flip_seed 2
run s60_gate 1831 --lr_gate
