#!/usr/bin/env bash
# Lab PC: memory of the b27 preset (26.5B parameters, 22.85B ternary) with TTF without the short momentum
# (--ts_nomom), ranks 1024, 12-bit accumulator, lean float tail, update inside the backward, every layer checkpointed,
# micro-batch 1 x 2048. Two placements of the state: on the GPU (gpu) and in page-locked host memory with async copies
# (host). 12 steps each; --mem_report; nvidia-smi sampled every second; memory guard as in b27_lean.sh.
# Summary ~/ternary-sync/b27_nomom.txt
#   bash scripts/lab/b27_nomom.sh [gpu host]   (RANK=1024 by default)
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
r=${RANK:-1024}; O=checkpoints/b27_nomom; mkdir -p $O; R=$HOME/ternary-sync/b27_nomom.txt
echo "$(date '+%F %T') start on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); RAM $(free -g | awk '/Mem/{print $2}') GB; ranks $r" >> $R
for where in ${*:-gpu host}; do
  OFF=""; [ $where = host ] && OFF="--ts_offload_async"
  n=r${r}_$where
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > $O/smi_$n.csv &
  SMI=$!
  /usr/bin/time -v python -m bitnet.train --preset b27 --mode kernel --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 1 --grad_accum 1 --steps 9155 --warmup 305 \
    --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense \
    --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 1000000 --eval_iters 1 --save_secs 100000000 --lookahead 0 --ckpt_skip 0 \
    --loss_chunk 256 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank $r --ts --ts_nomom --ts_theta 16 \
    --ts_tau 1000 --ts_anneal 0.5 --ts_fused --ts_store_a fp12s $OFF --lean_tail \
    --mem_report --stop_after 12 --out_dir $O/$n > $O/$n.log 2>&1 &
  PY=$!
  while kill -0 $PY 2>/dev/null; do
    [ "$(free -m | awk '/Mem/{print $7}')" -lt 2000 ] && { pkill -u $USER -f "out_dir $O/$n"; echo "$n: STOPPED by the memory guard" >> $R; }
    sleep 2
  done
  kill $SMI
  echo "$n: $(grep 'memory (MiB)' $O/$n.log | tail -1 | sed 's/  memory (MiB): //') | $(grep -E '^step ' $O/$n.log | tail -1 | cut -c1-100) | smi max $(sort -n $O/smi_$n.csv | tail -1) | host RSS $(grep 'Maximum resident' $O/$n.log | grep -oE '[0-9]+$' | awk '{printf "%.1f GB", $1/1048576}') $(grep -m1 -E 'Error|out of memory' $O/$n.log | cut -c1-120)" >> $R
done
echo "$(date '+%F %T') done" >> $R
