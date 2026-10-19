#!/bin/bash
# mix5b (19 Oct, after the GGUF ladder and the step-2349 benchmarks; user: knowledge matters less than knowing what it
# does not know and searching, and the biggest gap to fp is code): mix5 rebalanced, ~1B tokens. Reuses build_mix5.sh's
# streams in the same OUT directory and adds Toucan (MCP tool use, much of it web search), When2Call (answer / ask back /
# decline given the tools), OpenCodeInstruct (test-verified, more of it), OpenCodeReasoning (unread files) and Tulu IF.
# Shares: knowledge 20 (Wikipedia 11, FineWeb-Edu 9); code 25 (Nemotron code 10, OpenCodeInstruct 8, OpenCodeReasoning 7);
# agentic / search / tools 28 (Toucan 12, SWE-agent 8, OpenHands 1, Nemotron tool 3, Hermes 2, ToolACE 2); instruction
# following 8 (Tulu IF 6, Tulu general 2); knowing when not to 2 (When2Call); reasoning 15 (math 6, stem 4,
# OpenThoughts3 5); safety 2. Licences: CC-BY-4.0, MIT, Apache-2.0, ODC-BY.
#   MODEL=HF_DIR OUT=DIR bash scripts/quant/build_mix5b.sh      (after build_mix5.sh in the same OUT; CPU only)
set -u
M=${MODEL:?HF model dir}; D=${OUT:?output dir of build_mix5.sh}; P=${PY:-python}
export HF_HOME=${HF_HOME:-$D/hf_cache} HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
run() { nice -n 19 $P -u scripts/quant/sft_mix.py build $M $D/m5 --decon 13 "$@" > "$D/logs/b_$(echo "$*" | tr ' :-' '___').log" 2>&1; }
# each stream once (a restart after an interruption, e.g. the lab reboot, builds only what is missing; oci is rebuilt
# larger than mix5's, so its marker is a separate file)
[ -f $D/m5_toucan.bin ] || run --source toucan:130e6 &
[ -f $D/m5_when2call.bin ] || run --source when2call:20e6 &
[ -f $D/.oci5b ] || { run --source oci:88e6 --skip_files 4 && touch $D/.oci5b; } &
[ -f $D/m5_ocr.bin ] || run --source ocr:77e6 --skip_files 6 &
[ -f $D/m5_tulu_if.bin ] || run --source tulu:if:63e6 &
wait
ls -la $D/m5_*.bin
$P scripts/quant/sft_mix.py mix $D/mix5b_train.bin 1000e6 $D/m5_wiki.bin:0.11 $D/m5_fineweb.bin:0.09 \
  $D/m5_code.bin:0.10 $D/m5_oci.bin:0.08 $D/m5_ocr.bin:0.07 \
  $D/m5_toucan.bin:0.12 $D/m5_swe_agent.bin:0.08 $D/m5_openhands.bin:0.01 $D/m5_tool.bin:0.03 $D/m5_hermes.bin:0.02 \
  $D/m5_toolace.bin:0.02 $D/m5_tulu_if.bin:0.06 $D/m5_tulu_general.bin:0.02 $D/m5_when2call.bin:0.02 \
  $D/m5_math.bin:0.06 $D/m5_stem.bin:0.04 $D/m5_ot3.bin:0.05 $D/m5_tulu_safety.bin:0.02 --seed 6 && \
  sha256sum $D/mix5b_train.bin > $D/mix5b_train.bin.sha256 && echo MIX5B_DONE
