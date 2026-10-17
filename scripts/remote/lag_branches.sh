#!/usr/bin/env bash
# 4090: master's step-3000 state, 500 steps, trits following the latent only with probability q per step (--m_lag).
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
C=/scratch0/$USER/runs/curve_master/ckpt_3000.pt
for b in "lag10 --m_lag 0.1" "lag02 --m_lag 0.02"; do set -- $b; n=mbr_$1; shift; d=checkpoints/$n; rm -rf $d; mkdir -p $d; cp $C $d/ckpt.pt
  python -m ttf.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --lr 1.5e-3 \
    --min_lr 1.5e-4 --eval_interval 125 --eval_iters 30 --save_secs 1000000 --stop_after 3501 --track_flips \
    --resume --out_dir $d "$@" >> $d/train.log 2>&1
  rm -f $d/ckpt*.pt; touch $d/done
  echo "$(date '+%F %T') $n: $(grep -oE 'val loss [0-9.]+' $d/train.log | tr '\n' ' ')"; done
echo lag done
