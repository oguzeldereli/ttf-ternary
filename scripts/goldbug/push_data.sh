#!/usr/bin/env bash
# Laptop: copy the FineWeb-Edu bins (4 GB; too big for the 10 GB UCL home) straight to the machine's scratch.
#   bash scripts/goldbug/push_data.sh [HOST] [GB dir]     (default goldbug.cs.ucl.ac.uk, /scratch0/oguzelde/gb)
H=${1:-goldbug.cs.ucl.ac.uk}; G=${2:-/scratch0/oguzelde/gb}
cd "$(dirname "$0")/../.."
rsync -a --partial --progress -e "ssh -J oguzelde@knuckles.cs.ucl.ac.uk" data/fwedu32k_train.bin data/fwedu32k_val.bin \
  oguzelde@$H:$G/repo/data/
