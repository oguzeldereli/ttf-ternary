#!/bin/bash
# beachcomber (4090, 24 GB): Qwen3.8-27B-TTFT release run. GPTQ (streaming) -> trits frozen to 400 -> TTF theta 64.
# data: 50% Tulu-3 SFT chat (Qwen chat template, thinking off) + 50% Wikipedia, 64M tokens; val: wikiq8 (fp 1.9034 on 8 windows)
# usage: bash bc_q27.sh STEPS [--resume]
S=/tmp/oguzelde/tern; M=$S/hf/models/Qwen3.8-27B; N=qw27_rel; STEPS=$1; shift
source $S/env.sh; cd $S/repo; mkdir -p checkpoints/$N
export HF_HOME=$S/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
[ -f data/mixq8_train.bin ] || python scripts/quant/chat_data.py x data/chatq8 --mix data/wikiq8_train.bin data/mixq8_train.bin 0.5 --mix_total 64e6 >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') start STEPS=$STEPS $*" >> checkpoints/$N/train.log
python -m ttf.train --init_hf $M --init_hf_method gptq --init_stream \
  --data data/mixq8_train.bin --val data/wikiq8_val.bin --data_u32 --seq_len 2048 --batch_size 1 --grad_accum 16 --ts_micro \
  --steps $STEPS --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100 --eval_iters 4 --save_secs 3600 \
  --snap_every 0 --lookahead 0 --ckpt_skip 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 \
  --mode kernel --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 400 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal \
  --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag --ts_store_a fp12s --lean_tail \
  --mem_cap_gib 22.4 --mem_report --out_dir checkpoints/$N "$@" >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') exit $?" >> checkpoints/$N/train.log
