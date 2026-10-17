#!/usr/bin/env bash
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
O=/scratch0/$USER/runs/ts_smoke; rm -rf $O; mkdir -p $O
K="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 2
   --save_secs 1000000 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp
   --ts --stop_after 31"
for th in 4 16; do TERN_TIME=1 python -m ttf.train $K --ts_theta $th --out_dir $O/th$th > $O/th$th.log 2>&1
  echo "theta $th: $(grep -E '^step +(10|20|30) ' $O/th$th.log | grep -oE 'loss +[0-9.]+|flip [0-9.]+%' | tr '\n' ' ') | $(grep 'time step 30' $O/th$th.log | cut -d: -f2) | $(grep -m2 -E 'Error|error' $O/th$th.log)"; done
echo ts smoke done
