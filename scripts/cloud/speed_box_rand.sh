#!/bin/bash
# speed_box.sh without the 13 GB checkpoint and the data (boxes far from the lab, where the upload crawls): the 27B
# with random trits (--init_hf_method random), TTF from step 0, plain cross-entropy on the Wikipedia validation tokens
# (no teacher cache: the KD loss is ~1% of a step). Steps 0-1 compile and allocate the TTF state; step 3 is timed, a
# configuration named *_prof also profiles step 2 (TERN_PROF). The kernels see the shapes and rank of the real run, so
# the time per step is that of the real run up to the KD head and the flip counts.
#   bash speed_box_rand.sh "mb2|--batch_size 2 --grad_accum 8" "pp2_mb1|--batch_size 1 --grad_accum 16" ...
set -u
W=${W:-/workspace}; source $W/tern/env.sh; cd $W/tern/repo
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); CAP=$(python -c "print(round($MEM / 1024 - 1.2, 1))")
O=$W/tern/speed; mkdir -p $O; echo "$(date '+%F %T') $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x $(nvidia-smi -L | wc -l), cap $CAP GiB, random weights" | tee -a $O/summary.txt
F="--init_hf $W/tern/qw27 --init_hf_method random --data data/wikiq8_val.bin --val data/wikiq8_val.bin
  --data_u32 --seq_len 2048 --ts_micro --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100000
  --eval_iters 1 --save_secs 999999 --no_final_save --snap_every 0 --lookahead 0 --rc_scale --lowrank_mag add:16
  --mag_wd 0.1 --mode kernel --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 0 --ts_tau 1000 --ts_anneal 0.75
  --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag --ts_vdiag_every 50
  --ts_store_a fp12s --lean_tail --loss_chunk 512 --mem_cap_gib $CAP --mem_report"
for spec in "$@"; do
  N=${spec%%|*}; X=${spec#*|}; D=$O/$N; rm -rf $D ${D}_rank*; mkdir -p $D
  RUN="python -m bitnet.train"
  case $N in pp[0-9]*) G=${N#pp}; G=${G%%[!0-9]*}; RUN="python -m torch.distributed.run --nproc_per_node $G -m bitnet.train --pipe" ;; esac
  PR=; case $N in *_prof) PR=$D ;; esac
  TERN_TIME=1 TERN_PROF=$PR timeout 3000 $RUN $F $X --steps 4 --out_dir $D > $D/train.log 2>&1
  echo "$N [$X]: $(grep -o 'time step 3: forward+backward [0-9.]*s, update[^0-9]*[0-9.]*s' $D/train.log | tail -1) | peak $(grep -o 'peak in fwd+bwd [0-9.]*' $D/train.log | sort -t' ' -k4 -n | tail -1 | grep -o '[0-9.]*$') GiB | $(grep -c -i -E 'out of memory|Traceback' $D/train.log) errors" | tee -a $O/summary.txt
done
echo "$(date '+%F %T') done" | tee -a $O/summary.txt
