#!/bin/bash
# keep the last checkpoint saved before the trit release (step 400) as ckpt_pre400.pt (+ a copy of each as ckpt_sN.pt)
D=/scratch0/oguzelde/tern/repo/checkpoints/qw27_rel
last=""
while true; do
  n=$(grep -oE "checkpoint saved at step [0-9]+" $D/train.log | tail -1 | awk '{print $5}')
  if [ -n "$n" ] && [ "$n" != "$last" ]; then
    if [ "$n" -lt 400 ]; then
      sleep 20; cp $D/ckpt.pt $D/ckpt_s$n.pt.tmp && mv $D/ckpt_s$n.pt.tmp $D/ckpt_s$n.pt && ln -sfn ckpt_s$n.pt $D/ckpt_pre400.pt
      echo "$(date +%H:%M) kept step $n ($(stat -c %s $D/ckpt_s$n.pt) bytes)"
      last=$n
    else
      echo "$(date +%H:%M) step $n >= 400: done; pre-release checkpoint is $(readlink $D/ckpt_pre400.pt)"; exit 0
    fi
  fi
  sleep 60
done
