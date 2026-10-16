#!/bin/bash
# Lab PC: training-free GPTQ variants on the 110M fp model (scripts/analysis/gptq_variants.py): zero fraction, input
# rotation, centring, calibration size and damping. Output docs/analysis/geom110/gptq_variants.json (copied back by hand).
S=${S:-/tmp/$USER/tern}; source $S/env.sh || exit 1; cd $S/repo || exit 1; mkdir -p logs docs/analysis/geom110
V="plain:,cen:cen=1,zf15:zf=0.15,zf23:zf=0.23,zf30:zf=0.30,zf35:zf=0.35,zf23c:zf=0.23|cen=1,zf30c:zf=0.30|cen=1"
V="$V,rotdown:rot=down,rotout:rot=out,rotall:rot=all,rotoutc:rot=out|cen=1"
V="$V,calib8:calib=8,calib128:calib=128,calib512:calib=512,damp01:damp=0.1,damp1:damp=1.0"
python scripts/analysis/gptq_variants.py --fp $S/models/fp32_baseline_slim/ckpt.pt --variants "$V" \
  --out docs/analysis/geom110/gptq_variants.json
echo GPTQ_VARIANTS_DONE
