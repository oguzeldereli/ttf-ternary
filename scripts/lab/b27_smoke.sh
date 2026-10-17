#!/usr/bin/env bash
# Lab PC (16 GB 4070 Ti Super): the ~23B preset (b27: width 5120, 83 layers, 8 KV heads, MLP 13824; 22.85B ternary)
# trained with TTF on ONE GPU: fused update, real int8 / fp16 state (--ts_store_*), state in pinned host memory
# (--ts_offload), every layer checkpointed, micro-batch 1 x 2048. A few steps per rank; TERN_TIME prints time and GPU
# memory per step; nvidia-smi sampled every second; host RSS at the end. Summary ~/ternary-sync/b27_smoke.txt
#   bash scripts/lab/b27_smoke.sh [ranks...]     (default: 256 1024)
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
O=checkpoints/b27_smoke; mkdir -p $O; R=$HOME/ternary-sync/b27_smoke.txt
RANKS=${*:-256 1024}
echo "$(date '+%F %T') start on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); RAM $(free -g | awk '/Mem/{print $2}') GB" > $R
for r in $RANKS; do
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > $O/smi_r$r.csv &
  SMI=$!
  TERN_TIME=1 /usr/bin/time -v python -m ttf.train --preset b27 --mode kernel --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 1 --grad_accum 1 --steps 9155 --warmup 305 \
    --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --tail_fp32 \
    --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000000 --eval_iters 1 --save_secs 1000000 --lookahead 0 --ckpt_skip 0 \
    --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank $r --ts --ts_rank_s $r --ts_theta 16 --ts_tau 1000 \
    --ts_anneal 0.5 --ts_fused --ts_store_m int8 --ts_store_a fp16 --ts_offload --stop_after 5 \
    --out_dir $O/r$r > $O/r$r.log 2>&1
  kill $SMI
  echo "ranks $r: $(grep -E '^step ' $O/r$r.log | tail -1 | cut -c1-90)" >> $R
  grep -E "time step [0-9]+:" $O/r$r.log | tail -2 | sed 's/^/  /' >> $R
  grep -m2 "ts state" $O/r$r.log | sed 's/^/  /' >> $R
  echo "  nvidia-smi peak $(cut -d, -f2 $O/smi_r$r.csv | tr -dc '0-9\n' | sort -n | tail -1) MiB; host max RSS $(grep 'Maximum resident' $O/r$r.log | grep -oE '[0-9]+$' | awk '{printf "%.1f GiB", $1/1048576}')" >> $R
  grep -m3 -E "Error|error:|Killed|out of memory" $O/r$r.log | cut -c1-200 | sed 's/^/  /' >> $R
done
echo "$(date '+%F %T') done" >> $R
