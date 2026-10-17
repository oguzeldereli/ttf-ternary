#!/usr/bin/env bash
source /scratch0/oguzelde/env.sh; cd ~/ternary-LLM
O=/scratch0/oguzelde/runs/smoke; rm -rf $O; mkdir -p $O
M="--preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048
   --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 2
   --save_secs 1000000 --stop_after 21 --track_flips"
K="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 2
   --save_secs 1000000 --lookahead 0 --lowrank 1024 --ckpt_skip 2 --rc_scale --lr_gate --lr_vnorm 0.99 --lowrank_mag add:16
   --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 --spend 3 --stop_after 21"
t() { n=$1; shift; TERN_TIME=1 python -m ttf.train "$@" --out_dir $O/$n > $O/$n.log 2>&1
  echo "$n: $(grep -E '^step +20 ' $O/$n.log | cut -c1-40) | $(grep 'time step 20' $O/$n.log | cut -d: -f2) | $(grep -m1 -E 'Error|error' $O/$n.log)"; }
t plain $M
t adamx $M --m_beta1 0.9
t factv $M --m_factv
t rank512 $M --m_rank 512
t leak_snap $M --m_leak 300 --m_snap --m_clamp 1.5 --m_gfix 5
t vfull $K --lr_vfull
echo smoke done
