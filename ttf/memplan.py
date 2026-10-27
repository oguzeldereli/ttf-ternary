"""--ckpt-skip auto (the default): the activation checkpointing and the int8 trit cache chosen from measured memory.

Recomputing a block's forward costs as much as the forward (27B slice, laptop 4080: 0.97 s of a 9.97 s step for 2
blocks); keeping its activations costs memory (~1.1 GiB per 27B block at 1 x 2048 tokens). The int8 trit cache
(KernelTernaryLinear.w8) saves less per byte (27B: 0.06 s a step per 2 blocks for 1.7 GiB without recompute, against
0.45 s per GiB for the recompute), so it is taken only once every block keeps its activations.

The rule, per process (each pipeline stage / data-parallel rank on its own GPU), from the step's measured peak:
  budget = BUDGET x (the memory this process may use at the start: free + reserved, or --mem-cap-gib if lower)
  step 0 runs every block checkpointed and no trit cache; step 1 keeps the last block's activations, which measures
  d, the bytes one block adds; then each step keeps floor((budget - peak) / (MARGIN d)) more blocks, down again if a
  peak passed the budget, and the trit cache once all blocks are kept and its bytes fit. The choice changes how the
  backward gets its activations and where the trits come from, not a single number (same bits)."""
import math

import torch

BUDGET = 0.88      # of the usable memory (the allocator's fragmentation: 0.62 GiB at an 8.9 GiB peak on the 27B slice)
MARGIN = 1.25      # per block: the block measured (the stage's last) need not be the largest


class CkptAuto:
    def __init__(self, model, cap_gib=0.0, log=print):
        from . import layers as L
        self.model, self.log, self.L = model, log, L
        lo, hi = (model.pp["lo"], model.pp["hi"]) if getattr(model, "pp", None) else (0, len(model.layers))
        self.n = hi - lo
        self.pp = bool(getattr(model, "pp", None))
        free, _ = torch.cuda.mem_get_info()
        avail = free + torch.cuda.memory_reserved()
        if cap_gib:
            avail = min(avail, int(cap_gib * 2**30))
        self.budget = int(BUDGET * avail)
        self.w8_bytes = sum(2 * m.N * m.K for m in model.modules() if isinstance(m, L.KernelTernaryLinear))
        self.hist, self.d, self.done = [], None, False
        self._set(0)
        torch.cuda.reset_peak_memory_stats()

    def _set(self, k):
        self.skip = k
        self.model.ckpt_skip = k
        self.model._skip_set = False          # (QwenTern re-marks its layers on the next forward)

    def after_step(self):
        """at the end of a step: read its peak, choose the next step's checkpointing (returns the step's peak)"""
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self.hist.append((self.skip, self.L.W8_CACHE[0], peak))
        k, w8 = self.skip, self.L.W8_CACHE[0]
        if len(self.hist) >= 2:
            (k0, w0, p0), (k1, w1, p1) = self.hist[-2], self.hist[-1]
            if k1 != k0 and w1 == w0:
                self.d = max(self.d or 0, (p1 - p0) / (k1 - k0), 1)
        if peak > self.budget:                 # over: give back the trit cache first, then blocks
            if w8:
                self.L.W8_CACHE[0] = False; self._drop_w8()
            elif k > 0:
                self._set(max(0, k - max(1, math.ceil((peak - self.budget) / (self.d or peak)))))
        elif self.n and k == 0 and self.d is None:
            self._set(1)                       # measure d
        elif k < self.n and self.d:
            add = int((self.budget - peak) // (MARGIN * self.d))
            if self.pp:   # --pipe: a stage holds several micro-batches in flight, and the step-1 measure of d missed
                add = min(add, 2)   # some (27B stage 0: 1.38 GiB against 2.33): at most 2 blocks a step, each re-measured
            if add > 0:
                self._set(min(self.n, k + add))
        elif k == self.n and not w8 and peak + MARGIN * self.w8_bytes < self.budget and \
                not any(w for _, w, p in self.hist if w and p > self.budget):
            self.L.W8_CACHE[0] = True
        if (self.skip, self.L.W8_CACHE[0]) != (k, w8):
            self.log(f"--ckpt-skip auto: peak {peak / 2**30:.2f} of {self.budget / 2**30:.2f} GiB -> {self.skip} of "
                     f"{self.n} blocks without recompute{', int8 trit cache' if self.L.W8_CACHE[0] else ''}"
                     + (f" (a block: {self.d / 2**30:.2f} GiB)" if self.d else ""))
        return peak

    def _drop_w8(self):
        for m in self.model.modules():
            if isinstance(m, self.L.KernelTernaryLinear):
                m.__dict__.pop("_w8c", None)
