#!/usr/bin/env bash
# Phase 3b: batch sweep at 0-bit (stateless, 1.58-bit packed, no evidence array).
# Same 300M-token budget, same seed, same LR schedule, same flip rate; the ONLY
# variable is tokens-per-step (gradient accumulation 1/4/16/64).
# micro-batch 16 x seq 2048 = 32,768 tokens per micro-step.
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.triton/cache}
BUDGET=300000000
MICRO=32768
COMMON="--preset small --mode kernel --data data/wiki32k_train.bin \
--val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --lr 3e-4 --min_lr 3e-5 \
--eval_interval 1000 --eval_iters 30 --track_flips --int8 --dw_mode dense --save_secs 1800"
for ACC in 1 4 16 64; do
  TPS=$((MICRO * ACC))
  STEPS=$((BUDGET / TPS))
  WARMUP=$((STEPS / 30)); [ $WARMUP -lt 10 ] && WARMUP=10
  OUT=checkpoints/p3b_acc${ACC}
  echo "=== acc ${ACC}: ${TPS} tok/step, ${STEPS} steps, warmup ${WARMUP} -> ${OUT}"
  [ -f "${OUT}/done" ] && { echo "already done, skipping"; continue; }
  mkdir -p $OUT
  python -m ttf.train $COMMON --grad_accum $ACC --steps $STEPS --warmup $WARMUP \
      --out_dir $OUT > ${OUT}.log 2>&1 && touch ${OUT}/done
  grep -E "FINAL|Error|OutOfMemory" ${OUT}.log | tail -3
done
echo "SWEEP COMPLETE"
