#!/bin/bash
# Laptop side of the qw27_rel end sequence: pull the checkpoint, packed file, logs and results from beachcomber into
# models/qw27_rel/ and check the sha256 files written there by qw27_end.sh. Safe to rerun (rsync --partial; results
# that are still being written are pulled again next time).
set -u
cd "$(dirname "$0")/../.."
D=models/qw27_rel; mkdir -p $D
SSH="ssh -o BatchMode=yes -J oguzelde@knuckles.cs.ucl.ac.uk"
SRC=oguzelde@beachcomber.cs.ucl.ac.uk:/scratch0/oguzelde/tern/end/qw27_rel/
rsync -a --partial --info=progress2 -e "$SSH" "$SRC" $D/
(cd $D && for s in *.sha256; do sha256sum -c "$s" || echo "CHECKSUM FAILED: $s"; done)
ls -la $D
