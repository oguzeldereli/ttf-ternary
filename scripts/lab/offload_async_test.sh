#!/usr/bin/env bash
# Lab PC: --ts_offload_async against --ts_offload (copies on the compute stream) and the state on the GPU.
#   offload_async_test.sh eq     110M, 40 steps each: the loss lines must be identical (same seed; the copies only move)
#   offload_async_test.sh time   1.3B micro-batch 1 x 2048 (the lean_mem.sh setting), 30 steps each: s/step and peak
# Summary ~/ternary-sync/offload_async_MODE.txt
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
M=${1:-eq}; O=checkpoints/offload_async; mkdir -p $O; R=$HOME/ternary-sync/offload_async_$M.txt
B="--mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --grad_accum 1
   --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
   --dw_mode dense --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000000 --eval_iters 1 --save_secs 1000000 --lookahead 0
   --ckpt_skip 0 --loss_chunk 256 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_theta 16 --ts_tau 1000
   --ts_anneal 0.5 --ts_fused --ts_store_m int8 --ts_store_a fp16 --lean_tail --mem_report --lowrank 512 --ts_rank_s 512"
echo "$(date '+%F %T') start on $(hostname) $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)" > $R
if [ $M = eq ]; then P="--preset small --batch_size 2 --stop_after 41"; else P="--preset d2048_l24 --batch_size 1 --stop_after 31"; fi
for v in gpu sync async; do
  case $v in gpu) F="";; sync) F="--ts_offload";; async) F="--ts_offload_async";; esac
  rm -rf $O/${M}_$v
  TERN_TIME=1 python -m bitnet.train $(echo $B $P) $F --out_dir $O/${M}_$v > $O/${M}_$v.log 2>&1
  grep -E '^step ' $O/${M}_$v.log | sed -E 's/\| +[0-9.]+s \|.*//' > $O/${M}_$v.loss
  echo "$v: $(grep 'memory (MiB)' $O/${M}_$v.log | tail -1 | sed 's/  memory (MiB): //') | $(grep -E '^step ' $O/${M}_$v.log | awk -F'|' '{gsub(/[ s]/,"",$6); n=$1; sub(/step */,"",n); if (n+0==10) {a=$6; s0=n} b=$6; s1=n} END {printf "%.2f s/step (10-%d)", (b-a)/(s1-s0), s1}') | last $(tail -1 $O/${M}_$v.loss) $(grep -m1 -E 'Error|out of memory' $O/${M}_$v.log | cut -c1-120)" >> $R
done
echo "sync vs async loss lines: $(cmp -s $O/${M}_sync.loss $O/${M}_async.loss && echo IDENTICAL || echo DIFFER) ($(wc -l < $O/${M}_async.loss) lines); gpu vs sync: $(cmp -s $O/${M}_gpu.loss $O/${M}_sync.loss && echo IDENTICAL || echo DIFFER)" >> $R
echo "$(date '+%F %T') done" >> $R
