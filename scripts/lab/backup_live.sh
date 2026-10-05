#!/usr/bin/env bash
# Laptop: every hour, copy the latest checkpoint (+ train.log, metrics.jsonl) of every running lab run to
# checkpoints/live/RUN/, so a reboot or crash (lab /tmp is wiped by the Mon / Thu reboots) loses at most ~1.5 h.
# Runs listed as "RUN HOST" lines in the file given (default: the sweep scheduler's started list). rsync writes to a
# temporary file and renames, so the laptop copy is always a complete checkpoint.
#   bash scripts/lab/backup_live.sh [RUN-HOST list]        (restore: scripts/lab/restore_run.sh)
cd "$(dirname "$0")/../.."
LIST=${1:-$CLAUDE_JOB_DIR/tmp/sweep_started.txt}; LOG=checkpoints/live/backup.log; mkdir -p checkpoints/live
J="ssh -o BatchMode=yes -o ConnectTimeout=15 -J oguzelde@knuckles.cs.ucl.ac.uk"
while true; do
  while read -r r h; do
    [ -z "$r" ] || [ "${r:0:1}" = "#" ] && continue
    grep -q "FINAL val loss" checkpoints/live/$r/train.log 2>/dev/null && [ -f checkpoints/live/$r/ckpt.pt ] && continue
    mkdir -p checkpoints/live/$r
    if timeout 900 rsync -a -e "$J" oguzelde@$h-l.cs.ucl.ac.uk:/tmp/oguzelde/tern/repo/checkpoints/$r/{ckpt.pt,train.log,metrics.jsonl} \
         checkpoints/live/$r/ 2>/dev/null; then
      echo "$(date '+%F %T') $r ($h) step $(grep -E '^step' checkpoints/live/$r/train.log | tail -1 | awk '{print $2}')" >> $LOG
    else
      echo "$(date '+%F %T') $r ($h) FAILED (host down or no checkpoint yet)" >> $LOG
    fi
  done < "$LIST"
  sleep 3600
done
