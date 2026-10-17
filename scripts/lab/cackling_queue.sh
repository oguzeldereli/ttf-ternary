#!/usr/bin/env bash
# cackling: small step (1/4 of the flip rate) from scratch to 300M, then your design (gain 1) at 1/4 of the rate as a
# branch of nola_lab at 131M -> 300M. No look-ahead.
set -u
L=$HOME/ternary-LLM/scripts/lab
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
pgrep -u "$USER" -f "scripts/lab/sync_out.sh" >/dev/null || setsid nohup bash $L/sync_out.sh >/dev/null 2>&1 &
mkdir -p $S/bench_nola; [ -f $S/bench_nola/ckpt_4000.pt ] || cp $HOME/ternary-sync/inbox/bench_nola/ckpt_4000.pt $S/bench_nola/
N=small_step_s0; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.005 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) \
  --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> $HOME/ternary-sync/lab_queue.log
bash $L/mech_run.sh mech_user_q_b131 user branch 9156 --mech_gain 1 --rate_peak 0.005
