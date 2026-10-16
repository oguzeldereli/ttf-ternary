#!/bin/bash
# Rented box: the deep compute profile of the 27B continuation (16 Oct). Per configuration: resume the release
# checkpoint, steps 1350-1352: 1350 compiles, 1351 is the clean timing (TERN_TIME), 1352 is profiled (TERN_PROF: by
# operation, by named range, by launching Python line; A also writes a chrome trace). nvidia-smi samples utilisation,
# power and clocks every second throughout. Results in $W/tern/prof/<config>/ and prof/summary.txt.
#   CONFIGS="A B C D" bash prof_27b.sh        (A 4x4 cache, B 8x2 cache, C 16x1 cache, D 4x4 no cache)
set -u
W=${W:-/workspace}; source $W/tern/env.sh; cd $W/tern/repo
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); CAP=$(python -c "print(round($MEM / 1024 - 1.5, 1))")
P=$W/tern/prof; mkdir -p $P; NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(date '+%F %T') $NAME, mem cap $CAP GiB, $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" | tee -a $P/summary.txt
python -c "import torch, triton, transformers; print('torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__)" | tee -a $P/summary.txt
F="--init_hf $W/tern/qw27 --init_hf_method gptq --init_stream --resume --data data/mix3_train.bin --val data/wikiq8_val.bin
  --data_u32 --seq_len 2048 --ts_micro --kd_cache $W/tern/kd --kd_cache_start 1350 --distill_w 0.5 --kd_mode rest
  --steps 1353 --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100000 --eval_iters 4 --save_secs 999999
  --no_final_save --snap_every 0 --lookahead 0 --ckpt_skip 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --mode kernel
  --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 400 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom
  --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag --ts_vdiag_every 50 --ts_store_a fp12s --lean_tail
  --loss_chunk 512 --mem_cap_gib $CAP --mem_report"
for cfg in ${CONFIGS:-A B C D}; do
  case $cfg in
    A) MB=4; X="--w_cache int8"; TR=1 ;;
    B) MB=8; X="--w_cache int8"; TR= ;;
    C) MB=16; X="--w_cache int8"; TR= ;;
    D) MB=4; X=""; TR= ;;
  esac
  D=$P/$cfg; rm -rf $D; mkdir -p $D; ln -f $W/tern/ckpt/qw27_rel_ckpt.pt $D/ckpt.pt
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,memory.used \
    --format=csv -l 1 > $D/smi.csv & SMI=$!
  TERN_TIME=1 TERN_PROF=$D TERN_PROF_TRACE=$TR timeout 2400 python -m bitnet.train $F --batch_size $MB --grad_accum $((16 / MB)) $X \
    --out_dir $D > $D/train.log 2>&1
  kill $SMI
  t1=$(grep -o "time step 1351: forward+backward [0-9.]*s, update[^0-9]*[0-9.]*" $D/train.log | grep -o "[0-9.]*s\?$\|backward [0-9.]*" | tr '\n' ' ')
  echo "$cfg: micro-batch $MB x $((16 / MB)) $X | step 1351: $(grep -o 'time step 1351: .*' $D/train.log | cut -c1-120) | $(grep -o 'peakVRAM [0-9.]*GiB' $D/train.log | tail -1) | $(grep -c -i 'out of memory' $D/train.log) OOM" | tee -a $P/summary.txt
  rm -f $D/ckpt.pt
done
(cd $W/tern && tar czf prof_results.tgz prof) && echo "$(date '+%F %T') done: $W/tern/prof_results.tgz" | tee -a $P/summary.txt
