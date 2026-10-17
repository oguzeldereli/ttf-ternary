#!/usr/bin/env bash
# 4090: the recipe (dry + spend, rank 512) with the flip selection blended from the usual rule to "cheap" on a cosine.
#   bash ~/ternary-LLM/scripts/remote/start.sh queue_cheapcos.sh
set -u
n=cheapcos512_dryspend_s0; mkdir -p checkpoints/$n
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 \
  --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 \
  --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 \
  --save_secs 1800 --snap_every 1000 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lr_gate --lr_vnorm 0.99 \
  --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 --spend 3 --pshape cheap --pshape_sched cos \
  $([ -f checkpoints/$n/ckpt.pt ] && echo --resume) --out_dir checkpoints/$n >> checkpoints/$n/train.log 2>&1
echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log | tail -1)"
