#!/usr/bin/env bash
# Low-rank magnitude beyond the trits, branched at 131M (ckpt_4000), 600 steps, same batches as the fixes4000
# branches (baseline val 3.3088): additive W = bT + AB^T and multiplicative W = bT o (1 + AB^T), rank 4.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
run() {
  local n=$1; shift
  rm -rf checkpoints/$n; mkdir -p checkpoints/$n; ln -s $S/ckpt_4000.pt checkpoints/$n/ckpt.pt
  python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 \
    --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 \
    --min_lr 1.5e-4 --stop_after 4601 --eval_iters 30 --eval_interval 200 --save_secs 100000 --lookahead 2 \
    --lookahead_xbatch --lowrank 256 --ckpt_skip 2 --resume "$@" --out_dir checkpoints/$n > checkpoints/$n/train.log 2>&1
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
run fix4000_magadd4 --lowrank_mag add:4 & run fix4000_magmul4 --lowrank_mag mul:4 & wait
