#!/usr/bin/env bash
# Dry run of the goldbug toolkit on a lab PC (GB=/tmp/$USER/gbdry): setup (cu129 env), smoke test, both arms of
# arms_dry.txt under start.sh, a deliberate kill of dry_ttf mid-run (the watchdog must resume it from its checkpoint),
# final evaluations, log sync. Report: ~/ternary-sync/goldbug_dryrun.txt
#   GB=/tmp/$USER/gbdry bash ~/ternary-LLM/scripts/goldbug/dryrun.sh
set -u
export GB=${GB:-/tmp/$USER/gbdry}; R=$HOME/ternary-sync/goldbug_dryrun.txt
echo "$(date '+%F %T') dry run on $(hostname -s), GB=$GB" > $R
bash $HOME/ternary-LLM/scripts/goldbug/setup.sh >> $R 2>&1 || { echo "SETUP FAILED" >> $R; exit 1; }
source $GB/env.sh; cd $GB/repo
bash scripts/goldbug/smoke.sh scripts/goldbug/arms_dry.txt > /dev/null 2>&1; cat $GB/smoke.txt >> $R
rm -rf checkpoints/dry_ttf checkpoints/dry_master; : > $GB/queue.log
bash scripts/goldbug/start.sh scripts/goldbug/arms_dry.txt >> $R 2>&1
until grep -q "saved at step\|checkpoint saved" checkpoints/dry_ttf/train.log 2>/dev/null; do sleep 15; done
sleep 30; pkill -f "[o]ut_dir checkpoints/dry_ttf" && echo "$(date '+%F %T') killed dry_ttf (crash test)" >> $R
while [ ! -f checkpoints/dry_ttf/final_eval.json ] || [ ! -f checkpoints/dry_master/final_eval.json ]; do sleep 30; done
sleep 5; cat $GB/queue.log >> $R
grep -E "resum|loaded|start step" checkpoints/dry_ttf/train.log | head -3 >> $R
for n in dry_ttf dry_master; do echo "$n: $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log) | $(grep -oE '"wiki_loss": [0-9.]+|"fw_loss": [0-9.]+' checkpoints/$n/final_eval.json | paste -sd' ')" >> $R; done
bash -c "source $GB/env.sh; timeout 60 bash scripts/goldbug/sync_logs.sh" > /dev/null 2>&1
ls $HOME/ternary-sync/goldbug/ >> $R 2>&1
$GB/env/bin/tmux kill-session -t gb 2>/dev/null
echo "$(date '+%F %T') DRY RUN DONE" >> $R
