#!/usr/bin/env bash
# Later-checkpoint branch test (205M tokens, where M is noise-dominated): 300 steps from ckpt_6243 of
# the momentum + look-ahead run, identical batches (resume fast-forwards the data), reversals logged.
#   A  = M proposes, look-ahead x2 keeps (the run itself)      B  = g proposes, same rate
#   C1 = g proposes at 0.5 x rate                              C2 = g proposes at 0.35 x rate
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
IN=$HOME/ternary-sync/inbox/ckpt_6243.pt
[ -f $S/ckpt_6243.pt ] || { cp $IN $S/ckpt_6243.pt && rm -f $IN; }
ARGS="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine
  --rate 0.0 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --track_reversals
  --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 6544 --eval_iters 30 --eval_interval 100000
  --save_secs 100000 --lookahead 2 --lookahead_xbatch --ckpt_skip 2 --resume"
run() {  # name, rate_peak, extra flags
  mkdir -p checkpoints/$1; ln -sf $S/ckpt_6243.pt checkpoints/$1/ckpt.pt   # symlink: the .pt sync skips it
  python -m ttf.train $ARGS --rate_peak $2 --out_dir checkpoints/$1 ${3:-} > checkpoints/$1/train.log 2>&1
  rm -f checkpoints/$1/ckpt.pt          # the branch end state is not needed; keep only the metrics
  echo "$(date '+%F %T') DONE $1 $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$1/train.log)"
}
# two at a time: the GPU is launch-bound, so pairs run at nearly full speed each
run b6243_A 0.02 "--lowrank 256" & run b6243_B 0.02 & wait
run b6243_C1 0.01 & run b6243_C2 0.007 & wait
