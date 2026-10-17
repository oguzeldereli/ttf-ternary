#!/usr/bin/env bash
# Lab PC: --ts_micro with --ts_nomom (11 Oct) against the exact batch: 110M final recipe (rank 512), 600 steps, the same
# 16 x 2048 tokens per step as 16 x 1 (reference), 4 x 4 with the dense gradient summed (grad accumulation), and 4 x 4
# micro-batched (no dense gradient across micro-batches). Log: logs/micro_nomom.log
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
mkdir -p logs
for job in "mn_ref|--batch_size 16 --grad_accum 1" "mn_dense|--batch_size 4 --grad_accum 4" "mn_micro|--batch_size 4 --grad_accum 4 --ts_micro"; do
  N=${job%%|*}; F=${job#*|}; rm -rf checkpoints/$N; mkdir -p checkpoints/$N
  python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --steps 600 --warmup 60 --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100 --eval_iters 30 \
    --save_secs 99999 --snap_every 0 --no_final_save --lookahead 0 --ckpt_skip 2 --int8 --dw_mode dense --track_flips \
    --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 \
    --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 512 --ts_rank_s 512 $F --out_dir checkpoints/$N \
    >> checkpoints/$N/train.log 2>&1
  echo "$(date '+%F %T') $N $(grep -oE 'val loss [0-9.]+' checkpoints/$N/train.log | awk '{printf "%s ", $3}') | $(grep -oE 'peakVRAM [0-9.]+GiB' checkpoints/$N/train.log | tail -1) | $(grep -E '^step +590' checkpoints/$N/train.log | grep -oE '[0-9.]+s \|' )" >> logs/micro_nomom.log
done
