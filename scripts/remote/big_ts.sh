#!/usr/bin/env bash
# 4090: the best --ts setting (short momentum rank 512, accumulator rank 512, tau 1000) at 340M, after ts16full_tau1000_s0
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
while pgrep -u $USER -f "out_dir checkpoints/ts16full_tau1000_s0" >/dev/null; do sleep 60; done
n=big_ts16rs512tau1000_s0; mkdir -p checkpoints/$n
python -m ttf.train --preset d1024_l24 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 \
  --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 \
  --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 \
  --save_secs 1800 --snap_every 1000 --lookahead 0 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp \
  --lowrank 512 --ts --ts_theta 16 --ts_tau 1000 --ts_rank_s 512 $([ -f checkpoints/$n/ckpt.pt ] && echo --resume) \
  --out_dir checkpoints/$n >> checkpoints/$n/train.log 2>&1
echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log | tail -1)"
