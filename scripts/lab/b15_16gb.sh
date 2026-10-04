#!/usr/bin/env bash
# Lab PC (16 GB card): does the ~15B preset (b15: width 6144, 38 layers, MLP 16384) train on 16 GB at micro-batch 1 x 2048
# with the lean recipe (int8 / fp16 state on the host with async copies, update inside the backward, lean float tail,
# every layer checkpointed)? 20 steps per rank, --mem_report; summary ~/ternary-sync/b15_16gb.txt
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
O=checkpoints/b15_16gb; mkdir -p $O; R=$HOME/ternary-sync/b15_16gb.txt
B="--preset b15 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 1
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 1000000 --eval_iters 1 --save_secs 100000000
   --lookahead 0 --ckpt_skip 0 --loss_chunk 256 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_theta 16
   --ts_tau 1000 --ts_anneal 0.5 --ts_fused --ts_store_m int8 --ts_store_a fp16 --ts_offload_async --lean_tail --mem_report
   --stop_after 21"
echo "$(date '+%F %T') start on $(hostname) $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader); RAM $(free -g | awk '/Mem/{print $2}') GB" > $R
for r in 64 256; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > $O/smi_r$r.csv &
  SMI=$!
  /usr/bin/time -v python -m bitnet.train $(echo $B) --lowrank $r --ts_rank_s $r --out_dir $O/r$r > $O/r$r.log 2>&1
  kill $SMI
  echo "ranks $r: $(grep 'memory (MiB)' $O/r$r.log | tail -1 | sed 's/  memory (MiB): //') | $(grep -E '^step ' $O/r$r.log | tail -1 | cut -c1-100) | smi max $(sort -n $O/smi_r$r.csv | tail -1) | host RSS $(grep 'Maximum resident' $O/r$r.log | grep -oE '[0-9]+$' | awk '{printf "%.1f GB", $1/1048576}') $(grep -m1 -E 'Error|out of memory' $O/r$r.log | cut -c1-120)" >> $R
done
echo "$(date '+%F %T') done" >> $R
