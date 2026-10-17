#!/usr/bin/env bash
# Laptop: check the trainer, sync the code to the shared home, then start the three new training rules
# (select / adaptrate / multibeta, branches of nola_lab at 131M -> 300M, no look-ahead) each on its own idle lab
# 3090: goosander first, then the first idle PCs not already in use by us.
#   bash scripts/lab/launch_new_rules.sh
set -u
cd "$(dirname "$0")/../.."
K=oguzelde@knuckles.cs.ucl.ac.uk
python -c "import ast; ast.parse(open('ttf/train.py').read()); print('trainer syntax ok')" || exit 1
rsync -az --exclude checkpoints --exclude data --exclude __pycache__ ./ $K:ternary-LLM/ && echo "code synced"
ssh -o BatchMode=yes $K "mkdir -p ternary-sync/inbox/bench_nola" && \
  rsync -a checkpoints/nola_lab/ckpt_4000.pt $K:ternary-sync/inbox/bench_nola/ && echo "branch checkpoint shipped"
USED="cackling mallard mandarin shoveler barnacle"
CANDS="bufflehead ruddy eider pintail harlequin gressingham goosander aylesbury smew wigeon canada crested scaup scoter brent gadwall"
RULES=(select adaptrate multibeta)
n=0
for h in $CANDS; do
  [ $n -ge ${#RULES[@]} ] && break
  case " $USED " in *" $h "*) continue ;; esac
  st=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -J $K oguzelde@$h-l.cs.ucl.ac.uk \
       "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; pgrep -u oguzelde -f '[b]itnet.train' | wc -l" 2>/dev/null | tr '\n' ' ')
  mem=$(echo $st | awk '{print $1}'); ours=$(echo $st | awk '{print $2}')
  [ -z "$mem" ] && { echo "$h: unreachable"; continue; }
  if [ "$mem" -gt 2000 ] || [ "${ours:-0}" -gt 0 ]; then echo "$h: busy (${mem} MiB, ours ${ours})"; continue; fi
  r=${RULES[$n]}
  echo "$h: free -> $r"
  # the lab PCs' login shell is csh: send the commands to bash on stdin
  timeout 1500 ssh -o BatchMode=yes -J $K oguzelde@$h-l.cs.ucl.ac.uk bash -s <<EOF
bash ~/ternary-LLM/scripts/lab/setup.sh 2>&1 | tail -1
setsid nohup bash ~/ternary-LLM/scripts/lab/new_rules.sh $r > /tmp/oguzelde/tern/$r.out 2>&1 < /dev/null & disown
echo "  launched $r"
EOF
  n=$((n + 1))
done
echo "started $n of ${#RULES[@]} rules"
