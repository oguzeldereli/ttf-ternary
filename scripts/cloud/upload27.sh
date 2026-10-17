#!/bin/bash
# Laptop: fill a rented box for train27_box.sh (17 Oct). The big files (27B checkpoint 13.1 GB, mix4s 1.3 GB, the mix4s
# teacher shards gathered by gather_kd4s.sh, NKD of them) go from shoveler's staging copy (/tmp/oguzelde/stage) in 6 parallel 1 GiB-chunk streams (one TCP
# stream to a far box ran at ~5 MB/s, six at ~13); the laptop's key is forwarded (ssh agent), nothing is stored on the
# lab PC. The laptop sends code and the model config. Checksums are verified on the box.
#   bash scripts/cloud/upload_lab.sh IP PORT        (the box's direct SSH: vast.ai "PUBLIC_IPADDR" and VAST_TCP_PORT_22)
#   STREAMS=24 CHUNK_MB=256 bash ...                 (far boxes: each ssh stream is window-limited, ~0.5 MB/s to the US)
set -u
cd "$(dirname "$0")/../.."
IP=$1; PT=$2; W=${W:-/workspace}; R="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PT root@$IP"
[ -n "${SSH_AUTH_SOCK:-}" ] || { eval $(ssh-agent -s -t 14400) > /dev/null; ssh-add ~/.ssh/id_ed25519 < /dev/null; }
$R "mkdir -p $W/tern/repo/data $W/tern/qw27 $W/tern/ckpt $W/tern/kd4s" || exit 1
rsync -az -e "ssh -p $PT -o BatchMode=yes" --exclude __pycache__ ttf scripts root@$IP:$W/tern/repo/
rsync -az -e "ssh -p $PT -o BatchMode=yes" models/qw27/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja,merges.txt,vocab.json} root@$IP:$W/tern/qw27/
rsync -az -e "ssh -p $PT -o BatchMode=yes" data/wikiq8_val.bin root@$IP:$W/tern/repo/data/
echo "$(date +%T) code / config sent; big files from shoveler"
ssh -A -o BatchMode=yes -o ServerAliveInterval=30 -J oguzelde@knuckles.cs.ucl.ac.uk oguzelde@shoveler-l.cs.ucl.ac.uk "bash -s $IP $PT $W ${STREAMS:-6} ${CHUNK_MB:-1024} ${NKD:-600}" <<'EOS'
IP=$1; PT=$2; W=$3; P=$4; C=$5; NK=$6; cd /tmp/oguzelde/stage
B="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p $PT root@$IP"
send() { f=$1; d=$2; n=$(( ($(stat -L -c %s $f) + C * 1048576 - 1) / (C * 1048576) ))
  seq 0 $((n - 1)) | xargs -P $P -I{} bash -c "for t in 1 2 3 4 5 6; do dd if=$f bs=1M skip=\$(({} * $C)) count=$C status=none | ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PT root@$IP 'dd of=$d bs=1M seek=\$(({} * $C)) conv=notrunc status=none' && break; sleep \$t; done"
  $B "truncate -s $(stat -L -c %s $f) $d"; }
send qw27_rel_ckpt.pt $W/tern/ckpt/qw27_rel_ckpt.pt; echo "$(date +%T) checkpoint sent"
send /tmp/oguzelde/m4s/mix4s_train.bin $W/tern/repo/data/mix4s_train.bin; echo "$(date +%T) mix4s sent"
ls kd4s/ | sort | head -$NK | sed 's|^|kd4s/|' > /tmp/oguzelde/kd4s_send.txt
/opt/ucl/bin/rsync -a --files-from=/tmp/oguzelde/kd4s_send.txt -e "ssh -o BatchMode=yes -p $PT" . root@$IP:$W/tern/; echo "$(date +%T) $(wc -l < /tmp/oguzelde/kd4s_send.txt) shards sent"
$B "cd $W/tern && sha256sum ckpt/qw27_rel_ckpt.pt repo/data/mix4s_train.bin | cut -c1-16"
echo "expected 384adfa72c92d42d (checkpoint), $(sha256sum /tmp/oguzelde/m4s/mix4s_train.bin | cut -c1-16) (mix4s)"
EOS
