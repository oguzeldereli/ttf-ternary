#!/bin/bash
# Rented box: seconds per 32k-token step of the 27B continuation per configuration (2 steps each: 1350 compiles, 1351
# is the timing); a configuration named *_prof also profiles step 1352 (TERN_PROF: by op, by range, by launching line).
# Lines of $W/tern/speed/summary.txt appear as each finishes (pulled while the box runs).
# A name starting ppG (e.g. pp4_mb2) runs G pipeline stages (--pipe under torchrun, one per GPU).
#   bash speed_box.sh "mb2|--batch_size 2 --grad_accum 8" "mb4|--batch_size 4 --grad_accum 4" "pp4_mb2|--batch_size 2 --grad_accum 8" ...
set -u
W=${W:-/workspace}; source $W/tern/env.sh; cd $W/tern/repo
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); CAP=$(python -c "print(round($MEM / 1024 - 1.2, 1))")
O=$W/tern/speed; mkdir -p $O; echo "$(date '+%F %T') $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), cap $CAP GiB" | tee -a $O/summary.txt
F="--init_hf $W/tern/qw27 --init_hf_method gptq --init_stream --resume --data data/mix3_train.bin --val data/wikiq8_val.bin
  --data_u32 --seq_len 2048 --ts_micro --kd_cache $W/tern/kd --kd_cache_start 1350 --distill_w 0.5 --kd_mode rest
  --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100000 --eval_iters 4 --save_secs 999999
  --no_final_save --snap_every 0 --lookahead 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --mode kernel
  --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 400 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom
  --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag --ts_vdiag_every 50 --ts_store_a fp12s --lean_tail
  --loss_chunk 512 --mem_cap_gib $CAP --mem_report"
for spec in "$@"; do
  N=${spec%%|*}; X=${spec#*|}; D=$O/$N; rm -rf $D ${D}_rank*; mkdir -p $D; ln -f $W/tern/ckpt/qw27_rel_ckpt.pt $D/ckpt.pt
  RUN="python -m bitnet.train"
  case $N in pp[0-9]*) G=${N#pp}; G=${G%%[!0-9]*}; RUN="python -m torch.distributed.run --nproc_per_node $G -m bitnet.train --pipe" ;; esac
  ST=1352; PR=; case $N in *_prof) ST=1353; PR=$D ;; esac
  TERN_TIME=1 TERN_PROF=$PR timeout 2400 $RUN $F $X --steps $ST --out_dir $D > $D/train.log 2>&1
  rm -f $D/ckpt.pt
  echo "$N [$X]: $(grep -o 'time step 1351: forward+backward [0-9.]*s, update[^0-9]*[0-9.]*s' $D/train.log) | $(grep -o 'peak in fwd+bwd [0-9.]*' $D/train.log | tail -1) GiB | $(grep -c -i -E 'out of memory|Traceback' $D/train.log) errors" | tee -a $O/summary.txt
done
echo "$(date '+%F %T') done" | tee -a $O/summary.txt
