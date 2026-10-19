#!/bin/bash
# mix5: ~1B tokens for the 27B release's continuation after the mix4s steps (19 Oct): knowledge the Wikipedia validation
# showed slipping (Wikipedia from where mix3 stopped reading it, FineWeb-Edu files no mix read), agentic workflows
# (resolved SWE-agent runs, OpenHands runs, Hermes and ToolACE tool calling, a second epoch of Nemotron tool), and the
# reasoning / code / math / STEM / safety mix kept so it does not drift. Unread files of every source where any are left
# (Nemotron code / math / stem from file 28, OpenThoughts3 from 8, OpenCodeInstruct from 4); decontaminated as mix3.
# Shares: Wikipedia 22, FineWeb-Edu 20, SWE-agent 10, OpenHands 1, Hermes 3, ToolACE 3, Nemotron tool 5, code 10, math 6,
# stem 5, OpenThoughts3 6, OpenCodeInstruct 4, Tulu safety 3, Tulu general 2. Licences: CC-BY-4.0, MIT, Apache-2.0, ODC-BY.
#   MODEL=HF_DIR OUT=DIR WIKI=data/wikiq8_train.bin bash scripts/quant/build_mix5.sh     (from the repository root; CPU only)
set -u
M=${MODEL:?HF model dir}; D=${OUT:?output dir}; W=${WIKI:?Wikipedia token stream}; P=${PY:-python}
export HF_HOME=${HF_HOME:-$D/hf_cache} HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
mkdir -p $D/logs
# mix3 read the first 80M tokens of the Wikipedia stream (its mixer takes each source from the start): the rest
$P -c "import numpy as np; a = np.memmap('$W', dtype=np.uint32, mode='r'); a[80_000_000:].tofile('$D/m5_wiki.bin'); print(len(a) - 80_000_000)"
run() { nice -n 19 $P -u scripts/quant/sft_mix.py build $M $D/m5 --decon 13 "$@" > "$D/logs/$(echo "$*" | tr ' :-' '___').log" 2>&1; }
run --source fineweb:210e6 --skip_files 4 &
run --source swe_agent:105e6 &
run --source openhands:20e6 &
run --source hermes:40e6 &
run --source toolace:40e6 &
run --source nemotron:tool:55e6 --skip_files 4 --max_files 9 &
run --source nemotron:code:105e6 --skip_files 28 --max_files 12 &
run --source nemotron:math:63e6 --skip_files 28 --max_files 12 &
run --source nemotron:stem:53e6 --skip_files 28 --max_files 12 &
run --source ot3:63e6 --skip_files 8 &
run --source oci:42e6 --skip_files 4 &
run --source tulu:safety:40e6 &
run --source tulu:general:21e6 &
wait
ls -la $D/m5_*.bin
$P scripts/quant/sft_mix.py mix $D/mix5_train.bin 1000e6 $D/m5_wiki.bin:0.22 $D/m5_fineweb.bin:0.20 \
  $D/m5_swe_agent.bin:0.10 $D/m5_openhands.bin:0.01 $D/m5_hermes.bin:0.03 $D/m5_toolace.bin:0.03 $D/m5_tool.bin:0.05 \
  $D/m5_code.bin:0.10 $D/m5_math.bin:0.06 $D/m5_stem.bin:0.05 $D/m5_ot3.bin:0.06 $D/m5_oci.bin:0.04 \
  $D/m5_tulu_safety.bin:0.03 $D/m5_tulu_general.bin:0.02 --seed 5
sha256sum $D/mix5_train.bin > $D/mix5_train.bin.sha256
echo MIX5_DONE
