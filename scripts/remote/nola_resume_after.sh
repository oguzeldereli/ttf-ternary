#!/usr/bin/env bash
# Resume the rank-256 no-look-ahead full run once the from-scratch 20M set has finished (its last job is
# s20_magmul4), so nothing shares the GPU with it.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
until grep -q "DONE s20_magmul4" checkpoints/remote_queue.log 2>/dev/null; do sleep 60; done
# a new booking starts with an empty scratch: restore the checkpoint (inbox) and the run's log/metrics
# (home copy) so the resumed run appends to its history
N=lowrank256_nola_full
mkdir -p checkpoints/$N
[ -f checkpoints/$N/ckpt.pt ] || { cp $HOME/ternary-sync/inbox/nola_ckpt.pt checkpoints/$N/ckpt.pt && rm -f $HOME/ternary-sync/inbox/nola_ckpt.pt; }
for f in metrics.jsonl train.log; do
  [ -f checkpoints/$N/$f ] || cp $HOME/ternary-sync/runs/$N/$f checkpoints/$N/$f
done
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --resume --out_dir checkpoints/lowrank256_nola_full \
  >> checkpoints/lowrank256_nola_full/train.log 2>&1
echo "$(date '+%F %T') DONE lowrank256_nola_full $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/lowrank256_nola_full/train.log)"
