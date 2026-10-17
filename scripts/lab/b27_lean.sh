#!/usr/bin/env bash
# Lab PC (16 GB 4070 Ti Super, 30 GB RAM): the b27 preset (26.5B parameters, 22.85B ternary) with everything measured
# tonight: lean float tail, int8 / fp16 state in exactly sized page-locked host memory (_PinnedPool), async copies with
# prefetch (--ts_offload_async), update inside the backward, every layer checkpointed, micro-batch 1 x 2048, loss in
# chunks. 8 steps per rank; --mem_report; nvidia-smi sampled every second; a guard stops the run if the host's available
# memory falls below 2 GB (ranks 1024 filled barbel's RAM on 3 Oct). Summary ~/ternary-sync/b27_lean.txt
#   bash scripts/lab/b27_lean.sh [ranks...]     (default: 256 512; STORE_A=fp12s for the 12-bit accumulator)
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
A=${STORE_A:-fp16}; O=checkpoints/b27_lean_$A; mkdir -p $O; R=$HOME/ternary-sync/b27_lean_$A.txt
RANKS=${*:-256 512}
echo "$(date '+%F %T') start on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); RAM $(free -g | awk '/Mem/{print $2}') GB" > $R
for r in $RANKS; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > $O/smi_r$r.csv &
  SMI=$!
  /usr/bin/time -v python -m ttf.train --preset b27 --mode kernel --data data/wiki32k_train.bin \
    --val data/wiki32k_val.bin --seq_len 2048 --batch_size 1 --grad_accum 1 --steps 9155 --warmup 305 \
    --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense \
    --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 1000000 --eval_iters 1 --save_secs 100000000 --lookahead 0 --ckpt_skip 0 \
    --loss_chunk 256 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank $r --ts --ts_rank_s $r --ts_theta 16 \
    --ts_tau 1000 --ts_anneal 0.5 --ts_fused --ts_store_m int8 --ts_store_a $A --ts_offload_async --lean_tail \
    --mem_report --stop_after 12 --out_dir $O/r$r > $O/r$r.log 2>&1 &
  PY=$!
  while kill -0 $PY 2>/dev/null; do
    [ "$(free -m | awk '/Mem/{print $7}')" -lt 2000 ] && { pkill -u $USER -f "out_dir $O/r$r"; echo "ranks $r: STOPPED by the memory guard" >> $R; }
    sleep 2
  done
  kill $SMI
  echo "ranks $r: $(grep 'memory (MiB)' $O/r$r.log | tail -1 | sed 's/  memory (MiB): //') | $(grep -E '^step ' $O/r$r.log | tail -1 | cut -c1-100) | smi max $(sort -n $O/smi_r$r.csv | tail -1) | host RSS $(grep 'Maximum resident' $O/r$r.log | grep -oE '[0-9]+$' | awk '{printf "%.1f GB", $1/1048576}') $(grep -m1 -E 'Error|out of memory' $O/r$r.log | cut -c1-120)" >> $R
done
echo "$(date '+%F %T') done" >> $R
