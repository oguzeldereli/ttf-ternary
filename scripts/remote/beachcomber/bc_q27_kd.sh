#!/bin/bash
# beachcomber: qw27_rel resumed on the post-training mix with distillation from the precomputed teacher targets.
# usage: bash bc_q27_kd.sh STEPS KD_START   (KD_START = the step the cache was introduced at; keep it on later resumes)
S=/tmp/oguzelde/tern; M=$S/hf/models/Qwen3.8-27B; N=qw27_rel; STEPS=$1; KS=$2; shift 2
source $S/env.sh; cd $S/repo
export HF_HOME=$S/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
echo "$(date '+%F %T') resume STEPS=$STEPS kd_cache_start=$KS $*" >> checkpoints/$N/train.log
python -m bitnet.train --init_hf $M --init_hf_method gptq --init_stream --resume \
  --data data/mix2_train.bin --val data/wikiq8_val.bin --data_u32 --seq_len 2048 --batch_size 1 --grad_accum 16 --ts_micro \
  --kd_cache $S/kd27 --kd_cache_start $KS --distill_w 0.5 --kd_mode rest \
  --steps $STEPS --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100 --eval_iters 4 --save_secs 3600 \
  --snap_every 0 --lookahead 0 --ckpt_skip 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 \
  --mode kernel --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 400 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal \
  --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag --ts_store_a fp12s --lean_tail \
  --loss_chunk 512 --mem_cap_gib 22.4 --mem_report --out_dir checkpoints/$N "$@" >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') exit $?" >> checkpoints/$N/train.log
