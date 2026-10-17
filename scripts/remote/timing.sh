#!/usr/bin/env bash
# step-time split: forward+backward vs update, recipe fp32 / tf32 vs master, 340M and 1.3B
source /scratch0/oguzelde/env.sh; cd ~/ternary-LLM
O=/scratch0/oguzelde/runs/timing; mkdir -p $O
R="--mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
   --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 2
   --save_secs 1000000 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lr_gate --lr_vnorm 0.99 --lowrank_mag add:16
   --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 --spend 3 --stop_after 26"
M="--mode master --master_dtype fp32 --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 2
   --save_secs 1000000 --stop_after 26"
run() { n=$1; shift; TERN_TIME=1 timeout 1200 python -m ttf.train "$@" --out_dir $O/$n > $O/$n.log 2>&1
  python - $O/$n.log $n <<'PY'
import re, sys
t = [(float(a), float(b)) for a, b in re.findall(r"forward\+backward ([\d.]+)s, update \S+ \S+ \S+ \S+ ([\d.]+)s", open(sys.argv[1]).read())][5:]
pk = re.findall(r"peakVRAM ([\d.]+)GiB", open(sys.argv[1]).read())
err = [l for l in open(sys.argv[1]) if "Error" in l][:1]
if t: print(f"{sys.argv[2]:22s} {len(t)} steps: forward+backward {sum(a for a,_ in t)/len(t):.3f}s  update {sum(b for _,b in t)/len(t):.3f}s  peak {pk[-1] if pk else '?'} GiB")
else: print(sys.argv[2], "no timing", err)
PY
}
nvidia-smi --query-gpu=name --format=csv,noheader
run big_recipe_fp32 --preset d1024_l24 $R
run big_recipe_tf32 --preset d1024_l24 $R --tf32
run big_master      --preset d1024_l24 $M
run x1b_recipe_fp32 --preset d2048_l24 $R
run x1b_recipe_tf32 --preset d2048_l24 $R --tf32
echo timing done
