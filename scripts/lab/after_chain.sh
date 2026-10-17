#!/usr/bin/env bash
# Lab PC: start list_chain.sh JOBFILE once this PC's current chain and training run (with its evaluation) are done.
#   setsid nohup bash ~/ternary-LLM/scripts/lab/after_chain.sh ~/ternary-sync/JOBFILE &
while pgrep -u "$USER" -f "[l]ist_chain.sh|[r]ecover_run.sh|[m]icro_nomom_test.sh" >/dev/null \
      || pgrep -u "$USER" -f "[p]ython -m ttf.train|[f]inal_eval.py" >/dev/null; do sleep 120; done
exec bash ~/ternary-sync/list_chain.sh "$1"
