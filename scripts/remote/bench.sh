#!/usr/bin/env bash
# Step-time benchmark on the remote GPU: the look-ahead x2 screen config (no probes), 80 steps,
# before (commit 9c906bb, host syncs) and after (current code) the sync removal.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
rm -rf $S/old && mkdir -p $S/old && git -C $REPO archive 9c906bbdaf80da349b73e7a82e9c4bc16181c4c3 | tar -x -C $S/old
ARGS="--preset small --mode kernel --data $REPO/data/wiki32k_train.bin --val $REPO/data/wiki32k_val.bin
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips
  --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 80 --eval_iters 2 --eval_interval 100000
  --save_secs 100000 --lookahead 2 --lookahead_xbatch --ckpt_skip 2"
for tag in old new; do
  src=$REPO; [ $tag = old ] && src=$S/old
  mkdir -p $REPO/checkpoints/bench_$tag
  (cd $src && PYTHONPATH=$src python -m ttf.train $ARGS --out_dir $REPO/checkpoints/bench_$tag \
     > $REPO/checkpoints/bench_$tag/train.log 2>&1)
  python - $REPO/checkpoints/bench_$tag/train.log <<'P'
import re, sys
t = [(int(l.split()[1]), float(re.search(r"\|\s*([\d.]+)s \|", l).group(1)))
     for l in open(sys.argv[1]) if l.startswith("step")]
(a, ta), (b, tb) = t[3], t[-1]
print(sys.argv[1].split("/")[-2], f"{(tb - ta) / (b - a):.3f} s/step over steps {a}-{b}")
P
done
