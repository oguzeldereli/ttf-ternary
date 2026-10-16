#!/bin/bash
# One lab PC's share of the mix4s teacher cache (16 Oct; same teacher and format as teacher_lab.sh / mix3): the
# supplement's tokens from shoveler, shards claimed through ~/kd4s_claims, written to /tmp/oguzelde/kd4s. Run after
# teacher_lab.sh's environment and model exist on the PC.
#   setsid nohup bash ~/ternary-LLM/scripts/quant/teacher_lab4s.sh > /tmp/oguzelde/teacher_lab4s.log 2>&1 &
set -u
T=/tmp/oguzelde; mkdir -p $T/kd4s
S=$T/tern; $S/env/bin/python -c "import transformers, fla" 2>/dev/null || S=$T/tern2
P=$S/env/bin/python; M=$T/hf27
bash ~/ternary-LLM/scripts/lab/setup.sh > /dev/null 2>&1 || true            # code refresh only (env and model are kept)
[ -f $T/mix4s_train.bin ] || [ "$(hostname -s)" = shoveler-l ] || /opt/ucl/bin/rsync -a -e "ssh -o BatchMode=yes" \
  oguzelde@shoveler-l.cs.ucl.ac.uk:/tmp/oguzelde/m4s/mix4s_train.bin $T/
[ "$(hostname -s)" = shoveler-l ] && [ ! -f $T/mix4s_train.bin ] && ln -s $T/m4s/mix4s_train.bin $T/mix4s_train.bin
echo "$(date +%T) mix4s $(sha256sum $T/mix4s_train.bin | cut -c1-16) (a40d3a283e3820aa expected)"
cd $S/repo; export PYTHONPATH=$S/repo
$P scripts/quant/teacher_topk.py $M $T/mix4s_train.bin $T/kd4s --sequential --windows 0 --claim_dir ~/kd4s_claims
echo "$(date +%T) TEACHER4S_DONE"
