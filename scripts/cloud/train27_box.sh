#!/bin/bash
# Rented box: continue the 27B run (qw27_rel, step 1350) with the 17 Oct compute changes (--ts_factored 128,
# --ts_fire_fused, --lean_batched) on the mix4s teacher cache (kd4s shards from index 0 at step 1351). Saves every
# SAVE_SECS (the puller on the lab side copies each new checkpoint off, train27_pull.sh). Settings by environment:
#   STEPS (last step, default 2850), THETA (64), KD (distill weight, 0.5), LR (peak of the rewarm, 2.5e-4),
#   MINLR (5e-5), RW (rewarm steps, 50), MB (micro-batch 4 -> grad_accum 16 / MB), X (extra flags)
#   bash train27_box.sh            (after setup_box.sh and upload27.sh)
set -u
W=${W:-/workspace}; source $W/tern/env.sh; cd $W/tern/repo
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); CAP=$(python -c "print(round($MEM / 1024 - 1.2, 1))")
MB=${MB:-4}; GA=$(( 16 / MB )); D=$W/tern/run27; mkdir -p $D
[ -f $D/ckpt.pt ] || ln -f $W/tern/ckpt/qw27_rel_ckpt.pt $D/ckpt.pt
echo "$(date '+%F %T') $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), cap $CAP GiB, micro $MB x $GA" >> $D/box.log
TERN_TIME=${TIME_STEPS:-0} python -m bitnet.train --init_hf $W/tern/qw27 --init_hf_method gptq --init_stream --resume \
  --data data/mix4s_train.bin --val data/wikiq8_val.bin --data_u32 --seq_len 2048 --batch_size $MB --grad_accum $GA \
  --ts_micro --kd_cache $W/tern/kd4s --kd_cache_start 1351 --distill_w ${KD:-0.5} --kd_mode rest \
  --steps ${STEPS:-2850} --warmup 300 --lr ${LR:-2.5e-4} --min_lr ${MINLR:-5e-5} --lr_rewarm 1351:${RW:-50} \
  --tail_fp32 --eval_interval 100 --eval_iters 4 --save_secs ${SAVE_SECS:-3600} --snap_every 0 --lookahead 0 \
  --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --mode kernel --int8 --dw_mode dense --ts --ts_theta ${THETA:-64} \
  --ts_start 400 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 256 \
  --ts_rank_s 256 --ts_vdiag --ts_vdiag_every 50 --ts_store_a fp12s --lean_tail --lean_batched --loss_chunk 512 \
  --ts_factored 128 --ts_fire_fused --mem_cap_gib $CAP ${X:-} --out_dir $D >> $D/train.log 2>&1
echo "$(date '+%F %T') exit $? $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/box.log
