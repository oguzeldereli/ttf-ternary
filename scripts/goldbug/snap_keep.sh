#!/usr/bin/env bash
# goldbug: keep one checkpoint per 1000 steps of each running arm for later per-token analysis (scripts/analysis/
# token_dump.py), at no copy cost: train.py replaces ckpt.pt atomically (os.replace), so a hard link keeps the old file.
# A link is made when the saved checkpoint's own step (read from it, memory-mapped) has entered a new thousand.
#   bash scripts/goldbug/snap_keep.sh RUN [RUN ...]      (loops; run in a tmux window)
source "${GB:-/scratch0/$USER/gb}/env.sh"; cd "$GB/repo/checkpoints"
declare -A last ino
while true; do
  for r in "$@"; do
    [ -f $r/ckpt.pt ] || continue
    i=$(stat -c %i $r/ckpt.pt); [ "${ino[$r]:-}" = "$i" ] && continue; ino[$r]=$i
    ln -f $r/ckpt.pt $r/.snap_tmp.pt     # pin this save before reading it (the next save may replace ckpt.pt)
    s=$(python -c "import torch; print(torch.load('$r/.snap_tmp.pt', map_location='cpu', weights_only=False, mmap=True)['step'])" 2>/dev/null)
    [ -n "$s" ] || { rm -f $r/.snap_tmp.pt; unset "ino[$r]"; continue; }    # unreadable now: retry next round
    k=$(( s / 1000 ))
    if [ "${last[$r]:--1}" != "$k" ]; then
      ln -f $r/.snap_tmp.pt $r/snap_$s.pt && last[$r]=$k && echo "$(date '+%F %T') $r snap_$s.pt" >> $GB/snap_keep.log
    fi
    rm -f $r/.snap_tmp.pt
  done
  sleep 120
done
