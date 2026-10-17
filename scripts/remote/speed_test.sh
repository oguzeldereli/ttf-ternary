#!/usr/bin/env bash
# 4090, after qabf_s0: --gemm cublas / --ts_orth chol / both + fused, 200 steps each at 110M: same losses? faster?
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
while pgrep -u $USER -f "out_dir checkpoints/qabf_s0" >/dev/null; do sleep 60; done
O=/scratch0/$USER/runs/speed; rm -rf $O; mkdir -p $O
K="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100 --eval_iters 4
   --save_secs 1000000 --lookahead 0 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank 512
   --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5 --stop_after 201"
K=$(echo $K)
for v in base cublas chol all; do x=""; [ $v = cublas ] && x="--gemm cublas"; [ $v = chol ] && x="--ts_orth chol"
  [ $v = all ] && x="--gemm cublas --ts_orth chol --ts_fused"
  TERN_TIME=1 python -m ttf.train $K $x --out_dir $O/$v > $O/$v.log 2>&1
  echo "$v: $(grep 'val loss' $O/$v.log | tr '\n' ' ' | cut -c1-80) | $(grep 'time step 150' $O/$v.log | cut -c15-200) $(grep -m1 -E 'Error|error' $O/$v.log)"; done
echo speed done
