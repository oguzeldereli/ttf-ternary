#!/usr/bin/env bash
# goldbug: start every arm of the arms file (one line per arm: "GPU NAME train flags...") in its own tmux window,
# plus the log sync. Survives closing the terminal; do NOT log out of the desktop session (that ends the booking).
#   source $GB/env.sh; bash scripts/goldbug/start.sh [arms file]        attach: tmux attach -t gb
set -eu
source "${GB:-/scratch0/$USER/gb}/env.sh"; cd "$GB/repo"
A=${1:-scripts/goldbug/arms.txt}; T=$GB/env/bin/tmux
$T has-session -t gb 2>/dev/null || $T new-session -d -s gb "bash scripts/goldbug/sync_logs.sh"
while read -r g n flags; do
  [ -z "$g" ] || [ "${g:0:1}" = "#" ] && continue
  $T new-window -t gb: -n $n "bash scripts/goldbug/queue_gpu.sh $g $n $flags"
  echo "started $n on GPU $g"
done < $A
