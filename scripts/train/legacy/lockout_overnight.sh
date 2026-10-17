#!/usr/bin/env bash
# Per-weight flip lockout ("each weight flips once, then waits"):
#   1. screen three variants for 1000 steps (~33M tokens)
#   2. full 300M-token run of the best, comparable to armA_cosine (98.56)
# All arms keep the cosine flip-rate anneal, which is the one thing known to help.
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
LOG=checkpoints/lockout_overnight.log
say() { echo "[$(date +%m-%d\ %H:%M)] $*" | tee -a $LOG; }
final_ppl() { grep -oE "FINAL val loss [0-9.]+ \| ppl [0-9.]+" "$1" 2>/dev/null | tail -1 | awk '{print $NF}'; }

BASE="--preset small --mode kernel --data data/wiki32k_train.bin \
--val data/wiki32k_val.bin --lr 3e-4 --min_lr 3e-5 --eval_iters 30 --int8 \
--dw_mode dense --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 \
--warmup 305 --rate_schedule cosine --rate 0.02 --track_flips"

screen() {  # screen <name> <lockout args...>
  local name=$1; shift
  local OUT=checkpoints/lo_$name
  [ -f $OUT/done ] && { say "screen $name already done ($(final_ppl $OUT.log))"; return 0; }
  rm -rf $OUT; mkdir -p $OUT
  say "START screen $name"
  python -m ttf.train $BASE "$@" --stop_after 1000 --eval_interval 100000 \
      --save_secs 100000 --out_dir $OUT > $OUT.log 2>&1 && touch $OUT/done
  local LF=$(python - <<PY
import json
r=[json.loads(l) for l in open("$OUT/metrics.jsonl") if '"locked_frac"' in l]
print(f"{r[-1]['locked_frac']*100:.1f}%" if r else "n/a")
PY
)
  say "screen $name -> ppl $(final_ppl $OUT.log) | locked at epoch end $LF"
}

screen once_t200      --flip_lockout 200  --lockout_mode once
screen once_t1000     --flip_lockout 1000 --lockout_mode once
screen norev_t1000    --flip_lockout 1000 --lockout_mode noreversal

BEST=$(python - <<'PY'
import re, glob, os
def ppl(p):
    try:
        m = re.findall(r'FINAL val loss [0-9.]+ \| ppl ([0-9.]+)', open(p).read())
        return float(m[-1]) if m else 1e9
    except Exception: return 1e9
c = {os.path.basename(p)[3:-4]: ppl(p) for p in glob.glob("checkpoints/lo_*.log")}
print(min(c, key=c.get) if c else "once_t200")
PY
)
say "screens: $(for f in checkpoints/lo_*.log; do printf '%s=%s ' $(basename $f .log) $(final_ppl $f); done)| control armA_cosine@1000steps=188.3 -> best: $BEST"

case $BEST in
  once_t200)   ARGS="--flip_lockout 200  --lockout_mode once" ;;
  once_t1000)  ARGS="--flip_lockout 1000 --lockout_mode once" ;;
  norev_t1000) ARGS="--flip_lockout 1000 --lockout_mode noreversal" ;;
esac

OUT=checkpoints/armA_cos_lockout
mkdir -p $OUT
say "START armA_cos_lockout ($BEST, 300M tokens)"
for TRY in 1 2 3; do
  [ $TRY -gt 1 ] && say "RETRY $TRY (resuming)"
  python -m ttf.train $BASE $ARGS --eval_interval 1000 --save_secs 600 --resume \
      --out_dir $OUT >> $OUT.log 2>&1 && break
  sleep 30
done
if grep -q "^done$" $OUT.log; then
  touch $OUT/done
  say "DONE armA_cos_lockout -> ppl $(final_ppl $OUT.log)   (armA_cosine without lockout: 98.56)"
  python plot_run.py $OUT --title "armA_cos_lockout: $BEST" >> $LOG 2>&1
  python plot_compare.py checkpoints/armA_cosine $OUT --out checkpoints/cmp_lockout.png \
      --title "flip lockout vs none (cosine anneal, 300M tokens)" >> $LOG 2>&1
else
  say "FAILED armA_cos_lockout"; tail -5 $OUT.log | tee -a $LOG
fi
say "LOCKOUT OVERNIGHT COMPLETE"
