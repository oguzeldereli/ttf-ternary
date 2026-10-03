#!/usr/bin/env bash
# Lab PC: measured memory of lean TTF training (micro-batch 1 x 2048, every layer checkpointed, loss in 256-token chunks,
# update inside the backward, int8 / fp16 state, lean float tail; optionally state in host memory) at 110M / 340M / 1.3B.
# 60 steps each with --mem_report and TERN_TIME. Summary ~/ternary-sync/lean_mem.txt
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
O=checkpoints/lean_mem; mkdir -p $O; R=$HOME/ternary-sync/lean_mem.txt
B="--mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 1 --grad_accum 1
   --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
   --dw_mode dense --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000000 --eval_iters 1 --save_secs 1000000 --lookahead 0
   --ckpt_skip 0 --loss_chunk 256 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_theta 16 --ts_tau 1000
   --ts_anneal 0.5 --ts_fused --ts_store_m int8 --ts_store_a fp16 --lean_tail --mem_report --stop_after 61"
echo "$(date '+%F %T') start on $(hostname)" > $R
for cfg in "small 512" "small 128" "small 512 off" "d1024_l24 512" "d1024_l24 512 off" "d2048_l24 512" "d2048_l24 512 off"; do
  set -- $cfg; p=$1; r=$2; off=${3:-}; n=${p}_r${r}${off:+_off}
  TERN_TIME=1 /usr/bin/time -v python -m bitnet.train --preset $p $(echo $B) --lowrank $r --ts_rank_s $r ${off:+--ts_offload} \
    --out_dir $O/$n > $O/$n.log 2>&1
  echo "$n: $(grep 'memory (MiB)' $O/$n.log | tail -1 | sed 's/  memory (MiB): //') | $(grep -E 'time step 60:' $O/$n.log | grep -oE 'forward\+backward [0-9.]+s') | host RSS $(grep 'Maximum resident' $O/$n.log | grep -oE '[0-9]+$' | awk '{printf "%.0f MB", $1/1024}') $(grep -m1 -E 'Error|out of memory' $O/$n.log | cut -c1-120)" >> $R
done
echo "$(date '+%F %T') done" >> $R
