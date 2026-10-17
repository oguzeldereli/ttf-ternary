#!/bin/bash
# run several commands one after another (each a quoted string), in the repo copy of this PC
#   setsid nohup bash /tmp/oguzelde/tern/repo/scripts/lab/chain.sh "bash scripts/lab/geom_arms.sh A" "..." &
cd /tmp/oguzelde/tern/repo
for c in "$@"; do echo "$(date '+%F %T') chain: $c" >> /tmp/oguzelde/chain.log; bash -c "$c"; done
echo "$(date '+%F %T') chain done" >> /tmp/oguzelde/chain.log
