#!/usr/bin/env bash
# 4090: our rule (dry + spend, rank 512, plain selection) from the cheap-cosine run's step 3000 (25% cheap at that
# point), 500 steps with --track_reversals: how many of our flips undo an earlier flip (master: 96% at 3000-3500)?
source /scratch0/$USER/env.sh; cd ~/ternary-LLM
d=checkpoints/ours_rev3000; rm -rf $d; mkdir -p $d; cp checkpoints/cheapcos512_dryspend_s0/ckpt_3000.pt $d/ckpt.pt
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 \
  --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 \
  --min_lr 1.5e-4 --eval_interval 125 --eval_iters 30 --save_secs 1000000 --lookahead 0 --lowrank 512 --ckpt_skip 2 \
  --rc_scale --lr_gate --lr_vnorm 0.99 --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 \
  --spend 3 --track_reversals --stop_after 3501 --resume --out_dir $d >> $d/train.log 2>&1
rm -f $d/ckpt*.pt; echo "ours_rev done $(grep -oE 'val loss [0-9.]+' $d/train.log | tr '\n' ' ')"
