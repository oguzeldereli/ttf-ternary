#!/bin/bash
# mix4s (16 Oct): a ~320M-token supplement to mix3 aimed at what the 27B release lacks most against full precision
# (HumanEval 0.63 vs 0.98: wrong logic, not format; HellaSwag / ARC-C 81-82% of fp; GSM8K already 94%). New files of every
# source only (no overlap with mix3: Nemotron files from 16 on, mix3 used 4-15; OCR / OT3 / FineWeb-Edu skip the files
# mix3 read), plus OpenCodeInstruct (test-verified Python functions) and Tulu-3's instruction-following set. Every source
# decontaminated against the evaluation items (word 13-grams). Shares: OpenCodeInstruct 25, Nemotron code 12.5,
# OpenCodeReasoning 7.5 (code 45), FineWeb-Edu 27.5, Nemotron stem 7.5, OpenThoughts3 7.5, Nemotron math 5, Tulu IF 6.25
# (Nemotron tool has fewer than 16 files, so no new tool data; mix3 has 10% tool). Lab PC (CPU only, niced, beside the teacher job):
#   setsid nohup bash ~/ternary-LLM/scripts/quant/build_mix4s.sh > /tmp/oguzelde/mix4s.log 2>&1 &
set -u
S=${S:-/tmp/oguzelde/tern}; [ -x $S/env/bin/python ] || S=/tmp/oguzelde/tern2
P=$S/env/bin/python; M=${M:-/tmp/oguzelde/hf27}; D=${D:-/tmp/oguzelde/m4s}; R=$S/repo
mkdir -p $D/logs $R/data; rsync -a ~/ternary-LLM/scripts/quant/sft_mix.py $R/scripts/quant/; rsync -a ~/evaldata/bench ~/evaldata/zeroshot $R/data/   # the evaluation items for the decontamination
export HF_HOME=$D/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
cd $R
run() { nice -n 19 $P -u scripts/quant/sft_mix.py build $M $D/m4 --decon 13 "$@" > "$D/logs/$(echo "$*" | tr ' :-' '___').log" 2>&1; }
run --source oci:80e6 &
run --source nemotron:code:40e6 --skip_files 16 --max_files 12 &
run --source ocr:24e6 --skip_files 3 &
run --source fineweb:88e6 --skip_files 2 &
run --source nemotron:stem:24e6 --skip_files 16 --max_files 12 &
run --source ot3:24e6 --skip_files 4 &
run --source nemotron:math:16e6 --skip_files 16 --max_files 12 &
run --source tulu:if:20e6 &
wait
ls -la $D/m4_*.bin
$P scripts/quant/sft_mix.py mix $D/mix4s_train.bin 320e6 $D/m4_oci.bin:0.25 $D/m4_code.bin:0.125 $D/m4_ocr.bin:0.075 \
  $D/m4_fineweb.bin:0.275 $D/m4_stem.bin:0.075 $D/m4_ot3.bin:0.075 $D/m4_math.bin:0.05 $D/m4_tulu_if.bin:0.0625 --seed 4
sha256sum $D/mix4s_train.bin > $D/mix4s_train.bin.sha256
echo MIX4S_DONE
