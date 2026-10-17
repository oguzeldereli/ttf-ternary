#!/usr/bin/env bash
# Laptop, night of 27-28 Sep, after the toy mechanism runs (seed 1): (1) seed 2 of the three toy mechanism arms and
# of plain momentum (toy seeds vary a lot); (2) the user's design without the move correction from scratch
# (mech_user_g0_s0, to 131M), no look-ahead.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "scripts/toy/mech.sh" >/dev/null; do sleep 30; done
run() {
  local n=$1; shift
  STEPS=15000 TOY_TAG=_v256 scripts/toy/run.sh $n --lowrank 64 --seed 2 "$@"
  TOY_V=256 python -m scripts.toy.eval checkpoints/toy/$n kernel 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)" >> checkpoints/toy/queue.log
}
run mom_s2 &
run mom_mech_v1_s2 --mech v1 &
run mom_mech_user_s2 --mech user --mech_gain 1 &
run mom_mech_user_g0_s2 --mech user --mech_gain 0 &
wait
N=mech_user_g0_s0; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --mech user --mech_gain 0 --stop_after 4001 \
  $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
