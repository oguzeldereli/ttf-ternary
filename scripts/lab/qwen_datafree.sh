#!/bin/bash
# Lab PC: data-free calibration on Qwen3.5-0.8B vs domain calibration (after qwen_calib_domain.sh made the eval sets):
# self-generated text, uniform random tokens, no data (ALS); scored on wiki / chat / mix12.
S=${S:-/tmp/$USER/tern}; source $S/env.sh || exit 1; cd $S/repo || exit 1
M=$S/hf/models/Qwen3.5-0.8B-Base
[ -f data/selfqw.bin ] || python scripts/quant/qwen_selfgen.py $M data/selfqw.bin --n 64 --seq 2048 --batch 16
[ -f data/randqw.bin ] || python -c "import numpy as np; np.random.default_rng(4321).integers(0, 248044, 64 * 2049, dtype=np.uint32).tofile('data/randqw.bin')"
python scripts/quant/qwen_calib_domain.py $M --calib self=data/selfqw.bin,random=data/randqw.bin,nodata=als \
  --eval wiki=data/wikiqw_val.bin,chat=data/chatqw_val.bin,mix12=data/mixvalqw12.bin --n 12 --out docs/analysis/qwen08_calib_datafree.json
echo DATAFREE_QWEN_DONE
