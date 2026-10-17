#!/usr/bin/env bash
# Laptop: check the trainer, sync code, ship the branch checkpoints, then start one job per lab PC.
#   launch_list.sh "host|script args..." ...
set -u
cd "$(dirname "$0")/../.."
K=oguzelde@knuckles.cs.ucl.ac.uk
python -c "import ast; ast.parse(open('ttf/train.py').read()); print('trainer syntax ok')" || exit 1
rsync -az --exclude checkpoints --exclude data --exclude __pycache__ --exclude private ./ $K:ternary-LLM/ && echo "code synced"
ssh -o BatchMode=yes $K "mkdir -p ternary-sync/inbox/bench_nola ternary-sync/inbox/master4000"
rsync -a checkpoints/nola_lab/ckpt_4000.pt $K:ternary-sync/inbox/bench_nola/
python -c "
import json
open('/tmp/master_metrics_4000.jsonl','w').writelines(l for l in open('checkpoints/curve_master/metrics.jsonl') if json.loads(l).get('step',0)<=4000)"
rsync -a checkpoints/curve_master/ckpt_4000.pt $K:ternary-sync/inbox/master4000/ckpt_4000.pt
rsync -a /tmp/master_metrics_4000.jsonl $K:ternary-sync/inbox/master4000/metrics_4000.jsonl && echo "branch checkpoints shipped"
for job in "$@"; do
  h=${job%%|*}; cmd=${job#*|}
  echo "== $h: $cmd"
  timeout 1500 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -J $K oguzelde@$h-l.cs.ucl.ac.uk bash -s <<EOF
nvidia-smi --query-gpu=memory.used --format=csv,noheader
bash ~/ternary-LLM/scripts/lab/setup.sh 2>&1 | tail -1
setsid nohup bash ~/ternary-LLM/scripts/lab/$cmd > /tmp/oguzelde/tern/launch_\$(date +%s).out 2>&1 < /dev/null & disown
echo "  launched"
EOF
done
