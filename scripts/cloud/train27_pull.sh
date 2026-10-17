#!/bin/bash
# Laptop, beside a train27_box.sh run: every 10 min the run's logs to results/cloud/run27/; each new checkpoint (its
# mtime changed) to shoveler's staging copy /tmp/oguzelde/stage/run27/ (box -> lab directly, the laptop's key forwarded:
# 13 GB never crosses the laptop's link); stops when box.log says the run exited. Prints one line per event.
#   bash scripts/cloud/train27_pull.sh IP PORT
set -u
cd "$(dirname "$0")/../.."
IP=$1; PT=$2; W=${W:-/workspace}; O=results/cloud/run27; mkdir -p $O; last=""
[ -n "${SSH_AUTH_SOCK:-}" ] || { eval $(ssh-agent -s -t 86400) > /dev/null; ssh-add ~/.ssh/id_ed25519 < /dev/null; }
while true; do
  rsync -az -e "ssh -p $PT -o BatchMode=yes -o ConnectTimeout=20" root@$IP:$W/tern/run27/{train.log,metrics.jsonl,box.log} $O/ 2>/dev/null
  echo "$(date +%T) $(grep -E '^step' $O/train.log 2>/dev/null | tail -1 | cut -c1-90) $(grep -oE 'val loss [0-9.]+' $O/train.log 2>/dev/null | tail -1)"
  mt=$(ssh -p $PT -o BatchMode=yes -o ConnectTimeout=20 root@$IP "stat -c %Y $W/tern/run27/ckpt.pt" 2>/dev/null)
  if [ -n "$mt" ] && [ "$mt" != "$last" ] && [ -n "$last" ]; then
    ssh -A -o BatchMode=yes -J oguzelde@knuckles.cs.ucl.ac.uk oguzelde@shoveler-l.cs.ucl.ac.uk \
      "mkdir -p /tmp/oguzelde/stage/run27 && /opt/ucl/bin/rsync -a -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PT' root@$IP:$W/tern/run27/ckpt.pt /tmp/oguzelde/stage/run27/ckpt.pt && ls -la /tmp/oguzelde/stage/run27/ckpt.pt" \
      && echo "$(date +%T) checkpoint ($mt) copied to shoveler"
  fi
  [ -n "$mt" ] && last=$mt
  grep -q "exit" $O/box.log 2>/dev/null && { echo "$(date +%T) run exited: $(tail -1 $O/box.log)"; break; }
  sleep 600
done
