#!/usr/bin/env bash
# 4090: master's step-3000 state (curve_master) continued for 500 steps with one ingredient of master's update taken
# away (ttf/master_opt.py); val every 125 steps. Our rule lost master's whole lead (0.085) within 250 steps.
#   bash scripts/remote/master_branches.sh      (needs /scratch0/$USER/runs/curve_master/ckpt_3000.pt)
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
C=/scratch0/$USER/runs/curve_master/ckpt_3000.pt
br() { n=mbr_$1; shift; d=checkpoints/$n; [ -f $d/done ] && return; rm -rf $d; mkdir -p $d; cp $C $d/ckpt.pt
  python -m ttf.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --lr 1.5e-3 \
    --min_lr 1.5e-4 --eval_interval 125 --eval_iters 30 --save_secs 1000000 --stop_after 3501 --track_flips \
    --resume --out_dir $d "$@" >> $d/train.log 2>&1
  rm -f $d/ckpt*.pt; touch $d/done
  echo "$(date '+%F %T') $n: $(grep -oE 'val loss [0-9.]+' $d/train.log | tr '\n' ' ')"; }
br ctrl
br leak300 --m_leak 300
br snap --m_snap
br factv --m_factv
br rank512 --m_rank 512
br leak30 --m_leak 30
br clamp1 --m_clamp 1.0
br clamp06 --m_clamp 0.6
br gfix --m_gfix 3001
br b1997 --m_beta1 0.997
br rank128 --m_rank 128
br leak100 --m_leak 100
echo branches done
