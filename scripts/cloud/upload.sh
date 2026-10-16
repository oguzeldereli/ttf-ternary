#!/bin/bash
# Laptop -> rented box: code, the Qwen3.8 config + tokenizer, the 27B release checkpoint (13.1 GB), the mix3 tokens
# (3.2 GB), the Wikipedia validation windows, and the first teacher shards for the benchmark (fetched from a lab PC).
#   bash scripts/cloud/upload.sh "ssh -p PORT root@HOST" [N_SHARDS]
set -u
cd "$(dirname "$0")/../.."
SSH=$1; N=${2:-40}; HOSTSPEC=${SSH##* }; W=${W:-/workspace}
R="rsync -a --partial --info=progress2 -e"
SSHCMD="${SSH% *}"
$SSH "mkdir -p $W/tern/repo/data $W/tern/qw27 $W/tern/ckpt $W/tern/kd"
$R "$SSHCMD" --exclude __pycache__ bitnet scripts $HOSTSPEC:$W/tern/repo/
$R "$SSHCMD" models/qw27/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja,merges.txt,vocab.json} $HOSTSPEC:$W/tern/qw27/
$R "$SSHCMD" data/m3/mix3_train.bin data/wikiq8_val.bin $HOSTSPEC:$W/tern/repo/data/
$R "$SSHCMD" models/qw27_rel/qw27_rel_ckpt.pt models/qw27_rel/qw27_rel_ckpt.pt.sha256 $HOSTSPEC:$W/tern/ckpt/
K=data/kd3_bench; mkdir -p $K
J="-o BatchMode=yes -J oguzelde@knuckles.cs.ucl.ac.uk"
for i in $(seq 0 $((N - 1))); do f=$(printf "kd_%05d.npz" $i); [ -f $K/$f ] || echo $f; done > $K/need.txt
for h in pintail mallard harlequin gadwall cackling shoveler; do      # shards are spread over the teacher PCs
  [ -s $K/need.txt ] || break
  rsync -a -e "ssh $J" --files-from=$K/need.txt oguzelde@$h-l.cs.ucl.ac.uk:/tmp/oguzelde/kd3/ $K/ 2>/dev/null
  for i in $(seq 0 $((N - 1))); do f=$(printf "kd_%05d.npz" $i); [ -f $K/$f ] || echo $f; done > $K/need.txt
done
echo "shards for the benchmark: $(ls $K/kd_*.npz | wc -l) of $N"
$R "$SSHCMD" $K/kd_*.npz $HOSTSPEC:$W/tern/kd/
$SSH "cd $W/tern/ckpt && sha256sum -c qw27_rel_ckpt.pt.sha256 | sed 's#.*: #checkpoint #'"
