#!/usr/bin/env bash
# Lab PC: real compressed --ts state (--ts_store_*) against the simulated formats (--ts_qfmt_*) and fp32: 200 steps at
# 110M each; the losses of "sim" and "real" must be identical, the stored state smaller. Summary ~/ternary-sync/store_test.txt
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
O=checkpoints/store_test; mkdir -p $O; R=$HOME/ternary-sync/store_test.txt
K="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100 --eval_iters 4
   --save_secs 1000000 --lookahead 0 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank 512
   --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5 --ts_fused --stop_after 201"
echo "$(date '+%F %T') start on $(hostname)" > $R
for v in fp32 sim real; do x=""; [ $v = sim ] && x="--ts_qfmt_m int8 --ts_qfmt_a fp16"; [ $v = real ] && x="--ts_store_m int8 --ts_store_a fp16"
  python -m ttf.train $(echo $K) $x --out_dir $O/$v > $O/$v.log 2>&1
  echo "$v: $(grep 'val loss' $O/$v.log | cut -c1-28 | tr '\n' ' ') | $(grep -m1 'ts state' $O/$v.log) | $(grep -E '^step +200 ' $O/$v.log | grep -oE 'loss +[0-9.]+|peakVRAM [0-9.]+GiB' | tr '\n' ' ') $(grep -m1 Error $O/$v.log)" >> $R
done
echo "$(date '+%F %T') done" >> $R
