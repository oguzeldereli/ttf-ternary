#!/usr/bin/env bash
# Laptop: plain momentum at 1/8 of the flip rate (--rate_peak 0.0025), branch of nola_lab at 131M -> 300M, no
# look-ahead: the next point of the step-size sweep after 1/4 (small_step_b131, goosander).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
N=small_step8_b131; mkdir -p checkpoints/$N
if [ ! -f checkpoints/$N/ckpt.pt ]; then
  cp checkpoints/nola_lab/ckpt_4000.pt checkpoints/$N/ckpt.pt
  python -c "import json;open('checkpoints/$N/metrics.jsonl','w').writelines(l for l in open('checkpoints/nola_lab/metrics.jsonl') if json.loads(l).get('step',0)<=4000)"
  echo "[branch of nola_lab at step 4000 (131M), no look-ahead, flip rate 1/8]" > checkpoints/$N/train.log
fi
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.0025 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --resume --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
