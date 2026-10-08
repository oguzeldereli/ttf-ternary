#!/usr/bin/env bash
# goldbug: start the arms of the arms file in tmux, plus the log sync. One line per arm: "GPU NAME train flags...".
# Lines with the same GPU run at the same time (they share the card); a GPU field ending in "+" (e.g. "1+") runs that arm
# after the previous arm of that GPU has finished (same tmux window, queue_gpu.sh one after the other).
# Survives closing the terminal; do NOT log out of the desktop session (that ends the booking).
#   source $GB/env.sh; bash scripts/goldbug/start.sh [arms file]        attach: tmux attach -t gb
set -eu
source "${GB:-/scratch0/$USER/gb}/env.sh"; cd "$GB/repo"
A=${1:-scripts/goldbug/arms.txt}; T=$GB/env/bin/tmux
$T has-session -t gb 2>/dev/null || $T new-session -d -s gb "bash scripts/goldbug/sync_logs.sh"
names=(); cmds=(); last=()          # one tmux window per chain; last[GPU] = index of that GPU's latest chain
while read -r g n flags; do
  [ -z "$g" ] || [ "${g:0:1}" = "#" ] && continue
  q="bash scripts/goldbug/queue_gpu.sh ${g%+} $n $flags"
  if [ "${g: -1}" = "+" ] && [ -n "${last[${g%+}]:-}" ]; then
    i=${last[${g%+}]}; cmds[$i]="${cmds[$i]}; $q"; names[$i]="${names[$i]}+$n"
  else
    names+=("$n"); cmds+=("$q"); last[${g%+}]=$(( ${#cmds[@]} - 1 ))
  fi
done < $A
for i in "${!cmds[@]}"; do
  $T new-window -t gb: -n "${names[$i]}" "bash -c '${cmds[$i]}'"
  echo "started ${names[$i]}"
done
