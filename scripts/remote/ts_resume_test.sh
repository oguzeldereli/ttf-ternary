#!/usr/bin/env bash
# 4090: --ts resume (full state in the checkpoint) and gradient accumulation tests, 110M, ~10 min
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
O=/scratch0/$USER/runs/tstest; rm -rf $O; mkdir -p $O
K="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --grad_accum 1
   --steps 9155 --warmup 30 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
   --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 2
   --save_secs 1000000 --lookahead 0 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank 512
   --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5"
K=$(echo $K)
python -m ttf.train $K --batch_size 16 --stop_after 61 --out_dir $O/full > $O/full.log 2>&1
python -m ttf.train $K --batch_size 16 --stop_after 41 --out_dir $O/res > $O/res1.log 2>&1
python -m ttf.train $K --batch_size 16 --stop_after 61 --resume --out_dir $O/res > $O/res2.log 2>&1
python -m ttf.train $(echo $K | sed 's/--grad_accum 1/--grad_accum 2/') --batch_size 8 --stop_after 61 --out_dir $O/acc > $O/acc.log 2>&1
python -m ttf.train $(echo $K | sed 's/--grad_accum 1/--grad_accum 2/') --batch_size 8 --stop_after 61 --ts_fused --out_dir $O/accf > $O/accf.log 2>&1
for f in full res2 acc accf; do echo "$f: $(grep -E '^step +(50|60) ' $O/$f.log | grep -oE 'loss +[0-9.]+|flip [0-9.]+%' | tr '\n' ' ') $(grep -m1 -E 'resumed --ts|Error' $O/$f.log)"; done
grep -E "saved|ckpt" $O/res1.log | tail -n 2
echo tstest done
