#!/usr/bin/env bash
# 26.5B (b27) memory probe with the current recipe (no momentum, 12-bit accumulator, update inside the backward, lean
# float tail, micro-batch 1 x 2048, every layer checkpointed): 11 steps with --mem_report and --mem_snapshot at step
# 10 (every CUDA allocation with its Python stack; scripts/analysis/mem_snapshot.py attributes the peak).
#   WHERE=host|gpu|split KEEP=0.7 RANK=1024 CAP=0 bash scripts/lab/b27_memprobe.sh   (from the repo root, env sourced)
# On the lab PCs a guard stops the run if free host memory drops under 2 GB.
r=${RANK:-1024}; W=${WHERE:-host}; O=checkpoints/b27_memprobe; mkdir -p $O; n=r${r}_${W}${KEEP:+_k$KEEP}
OFF=""; [ $W = host ] && OFF="--ts_offload_async"; [ $W = split ] && OFF="--ts_offload_async --ts_offload_keep ${KEEP:-0.7}"
CAPF=""; [ "${CAP:-0}" != 0 ] && CAPF="--mem_cap_gib $CAP"
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > $O/smi_$n.csv &
SMI=$!
python -m bitnet.train --preset b27 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 \
  --batch_size 1 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 \
  --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 1000000 --eval_iters 1 \
  --save_secs 100000000 --lookahead 0 --ckpt_skip 0 --loss_chunk 256 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp \
  --lowrank $r --ts_rank_s $r --ts --ts_nomom --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_fused \
  --gemm int8mm --ts_store_a fp12s --lean_tail $OFF $CAPF --mem_report --mem_snapshot $O/$n.pickle:10 --stop_after 11 --no_final_save \
  --out_dir $O/$n > $O/$n.log 2>&1 &
PY=$!
while kill -0 $PY 2>/dev/null; do
  [ "$(free -m | awk '/Mem/{print $7}')" -lt 2000 ] && { kill $PY; echo "$n: STOPPED by the memory guard" >> $O/$n.log; }
  sleep 2
done
kill $SMI
echo "$n: $(grep 'memory (MiB)' $O/$n.log | tail -1) | $(grep -E '^step ' $O/$n.log | tail -1 | cut -c1-100) | smi max $(sort -n $O/smi_$n.csv | tail -1) $(grep -m1 -E 'Error|out of memory' $O/$n.log | cut -c1-200)"
[ -f $O/$n.pickle ] && python scripts/analysis/mem_snapshot.py $O/$n.pickle --top 25 > $O/$n.mem.txt 2>&1
