#!/usr/bin/env bash
# Laptop: copy goldbug's checkpoints (ckpt.pt, train.log, metrics, final eval) to checkpoints/goldbug/ (resumable).
# Run every few hours during the booking and once at the end (scratch is wiped when the booking ends).
#   bash scripts/goldbug/pull.sh [HOST] [GB dir]
H=${1:-goldbug.cs.ucl.ac.uk}; G=${2:-/scratch0/oguzelde/gb}
cd "$(dirname "$0")/../.."; mkdir -p checkpoints/goldbug
rsync -a --partial -e "ssh -J oguzelde@knuckles.cs.ucl.ac.uk" --exclude 'smoke/' --exclude '*.pt.tmp' \
  oguzelde@$H:$G/repo/checkpoints/ checkpoints/goldbug/
