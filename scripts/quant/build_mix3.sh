#!/bin/bash
# mix3: the post-training mixture for the 27B continuation (~800M tokens), built on beachcomber's CPUs (one process per
# source, niced; the trainer on the GPU is not touched). Every source is decontaminated against the evaluation items
# (word 13-grams, sft_mix.py --decon). Nemotron files after the first 4 of each split's shuffled order, so nothing
# overlaps mix2's training files or the held-out validation files. Wikipedia: wikiq8_train.bin (its val split is held out).
#   bash scripts/quant/build_mix3.sh      (from the code tree; writes $D/m3_*.bin, then $D/mix3_train.bin)
set -u
D=${D:-/scratch0/oguzelde/tern/m3}; P=${P:-/scratch0/oguzelde/tern/env/bin/python}
M=${M:-/scratch0/oguzelde/tern/hf/models/Qwen3.8-27B}; W=${W:-/scratch0/oguzelde/tern/repo/data/wikiq8_train.bin}
export HF_HOME=$D/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false   # never the 10 GB UCL home
mkdir -p $D/logs
run() { nice -n 19 $P -u scripts/quant/sft_mix.py build $M $D/m3 --decon 13 "$@" > $D/logs/$(echo "$*" | tr ' :' '__').log 2>&1; }
run --source nemotron:math:96e6 --skip_files 4 --max_files 12 &
run --source nemotron:code:112e6 --skip_files 4 --max_files 12 &
run --source nemotron:stem:48e6 --skip_files 4 --max_files 12 &
run --source nemotron:chat:48e6 --skip_files 2 --max_files 6 &
run --source nemotron:tool:80e6 --skip_files 4 --max_files 12 &
run --source tulu:general:80e6 &
run --source tulu:safety:40e6 &
run --source ot3:72e6 &
run --source ocr:64e6 &
run --source fineweb:80e6 &
wait
ls -la $D/m3_*.bin
# shares of ~800M tokens (user, 15 Oct: less Wikipedia, more tool / code / reasoning): Wikipedia 10, FineWeb-Edu 10,
# Tulu general 10, Tulu safety 5, Nemotron math 12 / code 14 / stem 6 / chat 6 / tool 10, OpenThoughts3 9,
# OpenCodeReasoning 8 (a source short of its share repeats, see sft_mix mix: here only the safety set, ~1.6x)
$P scripts/quant/sft_mix.py mix $D/mix3_train.bin 800e6 $W:0.10 $D/m3_fineweb.bin:0.10 $D/m3_tulu_general.bin:0.10 \
  $D/m3_tulu_safety.bin:0.05 $D/m3_math.bin:0.12 $D/m3_code.bin:0.14 $D/m3_stem.bin:0.06 $D/m3_chat.bin:0.06 \
  $D/m3_tool.bin:0.10 $D/m3_ot3.bin:0.09 $D/m3_ocr.bin:0.08 --seed 3
sha256sum $D/mix3_train.bin > $D/mix3_train.bin.sha256
echo MIX3_DONE
