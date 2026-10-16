#!/bin/bash
# Rented box: seconds per 32k-token step of the 27B continuation, resumed from the release checkpoint (step 1350), at
# micro-batches of 1 / 2 / 4 / 8 / 16 sequences (16 / 8 / 4 / 2 / 1 accumulation steps), then data parallel over all
# GPUs at the best of them. 21 steps each; the time per step is taken over steps 1360-1370 (after compilation). Nothing
# is saved. Cost per 1000 steps = price per GPU-hour * GPUs * s/step / 3.6.
#   bash bench_27b.sh            (after setup_box.sh and upload.sh)
set -u
W=${W:-/workspace}; source $W/tern/env.sh; cd $W/tern/repo
G=$(nvidia-smi -L | wc -l); MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CAP=$(python -c "print(round($MEM / 1024 - 1.5, 1))")
B=$W/tern/bench; mkdir -p $B; NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(date '+%F %T') $G x $NAME, $MEM MiB each, mem cap $CAP GiB" | tee -a $B/summary.txt
F="--init_hf $W/tern/qw27 --init_hf_method gptq --init_stream --resume --data data/mix3_train.bin --val data/wikiq8_val.bin
  --data_u32 --seq_len 2048 --ts_micro --kd_cache $W/tern/kd --kd_cache_start 1350 --distill_w 0.5 --kd_mode rest
  --steps 1371 --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100000 --eval_iters 4 --save_secs 999999
  --no_final_save --snap_every 0 --lookahead 0 --ckpt_skip 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --mode kernel
  --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 400 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom
  --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag --ts_store_a fp12s --lean_tail --loss_chunk 512
  --mem_cap_gib $CAP --mem_report"
rate() { awk '/^step +1360 /{a=$0} /^step +1370 /{b=$0} END{ if (a && b) { split(a, x, "|"); split(b, y, "|");
          for (i in x) if (x[i] ~ /s *$/ && x[i] !~ /lr/) ta=x[i]; for (i in y) if (y[i] ~ /s *$/ && y[i] !~ /lr/) tb=y[i];
          gsub(/[ s]/, "", ta); gsub(/[ s]/, "", tb); printf "%.1f", (tb - ta) / 10 } else printf "fail" }' $1; }
peak() { grep -o "peakVRAM [0-9.]*GiB" $1 | tail -1; }
best=""; bestt=1e9
for mb in 1 2 4 8 16; do
  D=$B/b$mb; rm -rf $D; mkdir -p $D; ln -f $W/tern/ckpt/qw27_rel_ckpt.pt $D/ckpt.pt
  timeout 3000 python -m bitnet.train $F --batch_size $mb --grad_accum $((16 / mb)) --out_dir $D > $D/train.log 2>&1
  t=$(rate $D/train.log); echo "1 GPU, micro-batch $mb x $((16 / mb)): $t s/step, $(peak $D/train.log) $(grep -c -i 'out of memory' $D/train.log) OOM" | tee -a $B/summary.txt
  [ "$t" = fail ] && break
  awk -v t=$t -v b=$bestt 'BEGIN{exit !(t < b)}' && { best=$mb; bestt=$t; }
done
if [ $G -gt 1 ] && [ -n "$best" ]; then
  D=$B/dp$G; rm -rf $D; mkdir -p $D; ln -f $W/tern/ckpt/qw27_rel_ckpt.pt $D/ckpt.pt
  ga=$((16 / best / G)); [ $ga -lt 1 ] && ga=1
  timeout 3000 torchrun --nproc_per_node $G -m bitnet.train $F --batch_size $best --grad_accum $ga --out_dir $D > $D/train.log 2>&1
  echo "$G GPUs data parallel, micro-batch $best x $ga each ($((best * ga * G)) sequences / step): $(rate $D/train.log) s/step" | tee -a $B/summary.txt
fi
echo "$(date '+%F %T') bench done" | tee -a $B/summary.txt
