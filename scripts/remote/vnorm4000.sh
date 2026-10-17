#!/usr/bin/env bash
# Fix 2: propose from M / sqrt(factored second moment), branched at 131M (ckpt_4000), 600 steps, same batches
# as the fixes4000 branches (baseline val 3.3088). Tiny smoke test first.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 256 --batch_size 2 --grad_accum 1 --steps 100 --warmup 5 --rate 0.02 --g_ref 3.0 --int8 --dw_mode dense \
  --tail_fp32 --eval_iters 1 --eval_interval 100000 --save_secs 100000 --lowrank 16 --lookahead 2 --lookahead_xbatch \
  --lr_vnorm 0.99 --stop_after 12 --out_dir $S/smoke_vn > $S/smoke_vn.log 2>&1 \
  || { tail -20 $S/smoke_vn.log | tee checkpoints/vnorm_smoke.err; exit 1; }
rm -rf $S/smoke_vn; echo "$(date '+%F %T') vnorm smoke ok"
n=fix4000_vnorm99
rm -rf checkpoints/$n; mkdir -p checkpoints/$n; ln -s $S/ckpt_4000.pt checkpoints/$n/ckpt.pt
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 \
  --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 \
  --min_lr 1.5e-4 --stop_after 4601 --eval_iters 30 --eval_interval 200 --save_secs 100000 --lookahead 2 \
  --lookahead_xbatch --lowrank 256 --ckpt_skip 2 --lr_vnorm 0.99 --resume --out_dir checkpoints/$n \
  > checkpoints/$n/train.log 2>&1
rm -f checkpoints/$n/ckpt.pt
echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
