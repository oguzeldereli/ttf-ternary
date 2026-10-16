#!/bin/bash
# Laptop: fill a rented box for prof_27b.sh / bench_27b.sh. The big files (27B checkpoint 13.1 GB, mix3 3.2 GB, 60
# teacher shards) go from shoveler's staging copy (/tmp/oguzelde/stage) in 6 parallel 1 GiB-chunk streams (one TCP
# stream to a far box ran at ~5 MB/s, six at ~13); the laptop's key is forwarded (ssh agent), nothing is stored on the
# lab PC. The laptop sends code and the model config. Checksums are verified on the box.
#   bash scripts/cloud/upload_lab.sh IP PORT        (the box's direct SSH: vast.ai "PUBLIC_IPADDR" and VAST_TCP_PORT_22)
set -u
cd "$(dirname "$0")/../.."
IP=$1; PT=$2; W=${W:-/workspace}; R="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PT root@$IP"
[ -n "${SSH_AUTH_SOCK:-}" ] || { eval $(ssh-agent -s -t 14400) > /dev/null; ssh-add ~/.ssh/id_ed25519 < /dev/null; }
$R "mkdir -p $W/tern/repo/data $W/tern/qw27 $W/tern/ckpt $W/tern/kd" || exit 1
rsync -az -e "ssh -p $PT -o BatchMode=yes" --exclude __pycache__ bitnet scripts root@$IP:$W/tern/repo/
rsync -az -e "ssh -p $PT -o BatchMode=yes" models/qw27/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja,merges.txt,vocab.json} root@$IP:$W/tern/qw27/
rsync -az -e "ssh -p $PT -o BatchMode=yes" data/wikiq8_val.bin root@$IP:$W/tern/repo/data/
echo "$(date +%T) code / config sent; big files from shoveler"
ssh -A -o BatchMode=yes -o ServerAliveInterval=30 -J oguzelde@knuckles.cs.ucl.ac.uk oguzelde@shoveler-l.cs.ucl.ac.uk "bash -s $IP $PT $W" <<'EOS'
IP=$1; PT=$2; W=$3; cd /tmp/oguzelde/stage
B="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p $PT root@$IP"
send() { f=$1; d=$2; n=$(( ($(stat -L -c %s $f) + 1073741823) / 1073741824 ))
  seq 0 $((n - 1)) | xargs -P 6 -I{} bash -c "dd if=$f bs=1M skip=\$(({} * 1024)) count=1024 status=none | ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PT root@$IP 'dd of=$d bs=1M seek=\$(({} * 1024)) conv=notrunc status=none'"
  $B "truncate -s $(stat -L -c %s $f) $d"; }
send qw27_rel_ckpt.pt $W/tern/ckpt/qw27_rel_ckpt.pt; echo "$(date +%T) checkpoint sent"
send mix3_train.bin $W/tern/repo/data/mix3_train.bin; echo "$(date +%T) mix3 sent"
/opt/ucl/bin/rsync -a -e "ssh -o BatchMode=yes -p $PT" kd/kd_*.npz root@$IP:$W/tern/kd/; echo "$(date +%T) $(ls kd/kd_*.npz | wc -l) shards sent"
$B "cd $W/tern && sha256sum ckpt/qw27_rel_ckpt.pt repo/data/mix3_train.bin | cut -c1-16"
echo "expected 384adfa72c92d42d (checkpoint), a9c5a6fe096aabd3 (mix3)"
EOS
