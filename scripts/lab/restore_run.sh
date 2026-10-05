#!/usr/bin/env bash
# Laptop: resume a lab run on HOST from its laptop backup (checkpoints/live/RUN/ckpt.pt): copies the checkpoint, its
# train.log and metrics.jsonl to HOST, writes HOST.list with the run's flags and starts the runner (scratch_run.sh adds
# --resume when ckpt.pt exists). FLAGS must be the run's original flags.
#   bash scripts/lab/restore_run.sh HOST RUN FLAGS...
cd "$(dirname "$0")/../.."
h=$1; r=$2; shift 2
J="ssh -o BatchMode=yes -o ConnectTimeout=15 -J oguzelde@knuckles.cs.ucl.ac.uk"
[ -f checkpoints/live/$r/ckpt.pt ] || { echo "no backup for $r"; exit 1; }
timeout 60 $J oguzelde@$h-l.cs.ucl.ac.uk "bash ~/ternary-LLM/scripts/lab/setup.sh > /dev/null 2>&1; mkdir -p /tmp/oguzelde/tern/repo/checkpoints/$r" || exit 1
timeout 900 rsync -a -e "$J" checkpoints/live/$r/ckpt.pt checkpoints/live/$r/train.log checkpoints/live/$r/metrics.jsonl \
  oguzelde@$h-l.cs.ucl.ac.uk:/tmp/oguzelde/tern/repo/checkpoints/$r/ || exit 1
timeout 60 $J oguzelde@$h-l.cs.ucl.ac.uk bash -s <<EOS
echo "$r $*" > ~/ternary-LLM/scripts/night/$h.list
setsid nohup bash ~/ternary-LLM/scripts/night/runner_lab.sh $h.list >/dev/null 2>&1 < /dev/null &
sleep 30; grep -E "resumed|^step" /tmp/oguzelde/tern/repo/checkpoints/$r/train.log | tail -2 | cut -c1-100
EOS
