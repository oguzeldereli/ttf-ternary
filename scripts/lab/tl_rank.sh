#!/bin/bash
# Does the TTF rank matter when converting a real pretrained model? (17 Oct; the 27B runs at rank 256 for memory)
# TinyLlama 1.1B (hidden 2048, FFN 5632), centred GPTQ start, trits moving from step 0, theta 64, the 27B's lr schedule,
# labels only, 50M tokens (1526 x 16 x 2048, as micro-batches 4 x 4 with --ts_micro: 16 GB cards); moves tracked. The 12 Oct test (r256 = r1024) had ~1% of trits moving.
#   bash /tmp/oguzelde/tern/repo/scripts/lab/tl_rank.sh RANK
set -u
R=$1; S=/tmp/oguzelde/tern; source $S/env.sh; cd $S/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$S/hf/models/TinyLlama-1.1B-intermediate-step-1431k-3T; D=checkpoints/tlr$R${SUF:-}; mkdir -p $D
F="--init_hf $M --init_hf_method gptqc --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048
   --batch_size ${BS:-4} --grad_accum ${GA:-4} --ts_micro --steps ${STEPS:-1526} --warmup 300 --lr 5e-4 --min_lr 5e-5 --tail_fp32
   --eval_interval 100 --eval_iters 10 --save_secs 99999 --snap_every 0 --no_final_save --lookahead 0 --ckpt_skip 0
   --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --mode kernel --int8 --dw_mode dense --track_flips --ts --ts_theta 64
   --ts_start 0 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm
   --lowrank $R --ts_rank_s $R"
echo "$(date '+%F %T') start tlr$R on $(hostname -s): $(echo $F)" >> $D/train.log
python -m bitnet.train $(echo $F) --out_dir $D >> $D/train.log 2>&1
echo "$(date '+%F %T') train end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
