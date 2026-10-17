#!/usr/bin/env bash
# After the 340M master: the recipe at full rank (rank 1024 = min side of every 340M matrix) on the 340M model.
#   bash ~/ternary-LLM/scripts/remote/start.sh queue_scale2.sh
set -u
while pgrep -u "$USER" -f "[o]ut_dir checkpoints/big_master" >/dev/null; do sleep 60; done
C="--preset d1024_l24 --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 9155 --warmup 305 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000"
n=big_dryspend_r1024_s0; mkdir -p checkpoints/$n
python -m ttf.train $C --mode kernel --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 \
  --dw_mode dense --track_flips --tail_fp32 --lookahead 0 --lowrank 1024 --ckpt_skip 2 --rc_scale --lr_gate --lr_vnorm 0.99 \
  --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 --spend 3 \
  $([ -f checkpoints/$n/ckpt.pt ] && echo --resume) --out_dir checkpoints/$n >> checkpoints/$n/train.log 2>&1
echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log | tail -1)"
