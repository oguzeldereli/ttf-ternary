#!/bin/bash
# shoveler (3090 Ti, the 27B data staged in /tmp/oguzelde/stage): seconds per 32k-token step of the 27B continuation for
# the current code, 2 steps (1350 compiles, 1351 is the timing), with the teacher job stopped meanwhile and restarted after.
#   bash ~/ternary-LLM/scripts/lab/speed_27b.sh NAME "extra flags"     (default micro-batch 2 x 8)
set -u
N=$1; X=${2:-}; T=/tmp/oguzelde; S=$T/tern; [ -x $S/env/bin/python ] || S=$T/tern2
source $S/env.sh; R=$T/speed/code; mkdir -p $R $T/speed/$N; rsync -a --delete --exclude __pycache__ ~/ternary-LLM/bitnet ~/ternary-LLM/scripts $R/
cd $R; export PYTHONPATH=$R HF_HUB_OFFLINE=1
D=$T/speed/$N; ln -sf $T/stage/qw27_rel_ckpt.pt $D/ckpt.pt
for p in $(pgrep -f "[t]eacher_lab4s[.]sh"); do kill $p; done; for p in $(pgrep -f "[t]eacher_topk[.]py"); do kill $p; done
sleep 15     # the teacher holds ~8.6 GB: it is stopped for the test (its unfinished shard stays claimed: teacher_gaps.sh)
TERN_TIME=1 ${PROF:+TERN_PROF=$D} timeout 3000 python -m bitnet.train --init_hf $T/hf27 --init_hf_method gptq --init_stream --resume \
  --data $T/stage/mix3_train.bin --val $T/stage/mix3_train.bin --data_u32 --seq_len 2048 --batch_size 2 --grad_accum 8 --ts_micro \
  --kd_cache $T/stage/kd --kd_cache_start 1350 --distill_w 0.5 --kd_mode rest --steps ${STEPS:-1352} --warmup 300 --tail_fp32 \
  --lr 5e-4 --min_lr 5e-5 --eval_interval 100000 --eval_iters 4 --save_secs 999999 --no_final_save --snap_every 0 --lookahead 0 \
  --ckpt_skip 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --mode kernel --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 400 \
  --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag \
  --ts_vdiag_every 50 --ts_store_a fp12s --lean_tail --loss_chunk 512 --mem_cap_gib 22.4 --mem_report $X --out_dir $D > $D/train.log 2>&1
setsid nohup bash ~/ternary-LLM/scripts/quant/teacher_lab4s.sh >> $T/teacher_lab4s.log 2>&1 < /dev/null &
echo "$N: $(grep -o 'time step 1351: .*' $D/train.log | cut -c1-110) | $(grep -o 'peak in fwd+bwd [0-9.]*' $D/train.log | tail -1) | $(grep -c -i -E 'out of memory|Traceback' $D/train.log) errors" | tee -a $T/speed/summary.txt
