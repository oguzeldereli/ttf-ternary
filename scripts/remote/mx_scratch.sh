#!/usr/bin/env bash
# 4090, after after_branches.sh: master from scratch (110M, 300M tokens) with a long first moment / rate-limited firing.
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
until grep -q "undo bench done" /scratch0/$USER/runs/after_branches.txt 2>/dev/null; do sleep 60; done
for b in "mx_lag02 --m_lag 0.02" "mx_b1997 --m_beta1 0.997"; do set -- $b; n=$1; shift; d=checkpoints/$n; mkdir -p $d
  python -m ttf.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --lr 1.5e-3 \
    --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 --track_flips \
    $([ -f $d/ckpt.pt ] && echo --resume) --out_dir $d "$@" >> $d/train.log 2>&1
  echo "$(date '+%F %T') $n $(grep -oE 'FINAL val loss [0-9.]+' $d/train.log | tail -1)"; done
