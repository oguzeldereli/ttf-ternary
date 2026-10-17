# Related work (first pass, 30 Sep 2026)

What is closest to this project, what each keeps in memory per weight during training, and what differs. Our method:
ternary weights are the only per-weight state (1.6 bit packed, no latent copy); the optimizer state is a rank-r momentum,
row/column second moments and a few scalars, $O(r(N+K))$ per layer ([FORMULAS.md](FORMULAS.md)). Best at 110M / 300M
tokens of Wikipedia: 2.7998 (ppl 16.4) vs master weights 2.751 (15.7).

## Closest: training quantized weights without a full-precision master copy

| work | what it does | per-weight training state | same as ours | different from ours |
|---|---|---|---|---|
| **DQT**, Direct Quantized Training with stochastic rounding (Zhao et al., ACML 2025) [[arXiv 2412.04787]](https://arxiv.org/abs/2412.04787) | updates the low-bit weights directly with stochastic rounding, no STE latent copy; LLaMA-style 130M / 320M on Wikipedia, 1B on Wikipedia and FineWeb | quantized weight + **AdamW moments per weight** | no master weights; ternary tested; **same data family (English Wikipedia) and sizes** | full-size Adam state (linear memory); ternary is feasible but 8 bits are needed to match BitNet b1.58 (1B FineWeb ppl: FP32 19.99, BitNet 28.20, DQT 8-bit 25.43) |
| **ECO**, quantized training without full-precision master weights (Nikdan, Zandieh, Alistarh, Mirrokni, ICML 2026) [[arXiv 2601.22101]](https://arxiv.org/abs/2601.22101) | applies updates to quantized weights and injects the quantization error into the optimizer momentum (error feedback); 30-800M, Gemma-3 1B, 2.1B MoE (FP8), DeepSeek-MoE-16B fine-tuning (INT4) | FP8/INT4 weight + **full-size momentum** | no master weights; the momentum carries what rounding drops (cf. our spend) | FP8/INT4, not ternary; per-weight momentum (linear memory); near-lossless vs master |
| **Bop**, "Latent weights do not exist" (Helwegen et al., NeurIPS 2019) [[arXiv 1906.02107]](https://arxiv.org/abs/1906.02107); second-order follow-up (Suarez-Ramirez et al., CVPRW 2021) [[arXiv 2104.05124]](https://arxiv.org/abs/2104.05124) | binary networks trained by flips: an exponential moving average of the gradient per weight, a weight flips when it passes a threshold | binary weight + **per-weight momentum** | latent-free flips driven by an accumulated gradient; the inertia argument (a stronger reverse signal is needed to flip back) | binary CNNs, not LLMs; full-size momentum; deterministic threshold instead of our stochastic, rate-scheduled flips |
| **Q-GaLore** (Zhang et al., 2024/25) [[arXiv 2407.08296]](https://arxiv.org/abs/2407.08296) | GaLore with INT4 projection matrices and **INT8 weights updated by stochastic rounding** (no full-precision weights); LLaMA-7B from scratch in 16 GB | INT8 weight + low-rank optimizer state | the closest in spirit: quantized weights without a master copy **and** low-rank optimizer state | 8-bit weights, not ternary; Adam in a projected subspace; stochastic rounding of real-valued updates rather than flips |
| **Low-Rank Ternary Adaptation** (Manolache, Li, van Gemert, ECCV 2026) [[arXiv 2608.24469]](https://arxiv.org/abs/2608.24469) | fine-tunes ternary models without dequantizing, via a low-rank Kronecker factorization of ternary multiplicative updates (sign flips, zeroing); ternarized LLaMA-3 1B / 3B, ViT | ternary weights + small adapters | ternary domain kept, discrete updates, low-rank structure | fine-tuning of already-trained ternary models, not pretraining from scratch |

## Low-rank or factored optimizer state (full-precision weights)

| work | idea | relation |
|---|---|---|
| **GaLore** (Zhao et al., ICML 2024) [[arXiv 2403.03507]](https://arxiv.org/abs/2403.03507) | project gradients to a low-rank subspace, run Adam there, project back; LLaMA 1B / 7B on C4 | low-rank optimizer state, but full-precision weights |
| **APOLLO** (MLSys 2025) [[GitHub]](https://github.com/gss10282023/APOLLO) | channel-wise gradient scaling estimated in a random low-rank space: SGD-like memory, AdamW-level quality | low-memory scaling of the update (cf. our factored Adam step) |
| **MoFaSGD**, low-rank momentum factorization (Mahdavinia, Mahdavi, 2025) [[arXiv 2507.08091]](https://arxiv.org/abs/2507.08091) | a dynamically updated low-rank SVD of the first-order momentum, spectrally normalized updates; fine-tuning | **the same object as our momentum** (low-rank factorized momentum updated each step), with full-precision weights |
| **Adafactor** (Shazeer and Stern, ICML 2018) | factored (row x column) second moment | our `--lr_vnorm` is this factorization, used to rescale the flip signal |

## Ternary models (with latent weights): the reference

| work | relation |
|---|---|
| **BitNet b1.58** (Ma et al., 2024) | the master-weights reference we compare to: absmean ternary forward, STE, AdamW on a latent copy; reports matching full precision from ~3B |
| **Spectra / TriLM** (Kaushal et al., ICLR 2025) [[arXiv 2407.12327]](https://arxiv.org/abs/2407.12327) | ternary LMs 99M-3.9B on 300B tokens with latent weights; 3.9B TriLM matches the 3.9B FloatLM |
| **TernaryLM** (2026) [[arXiv 2602.07374]](https://arxiv.org/abs/2602.07374) | native ternary 132M model |

## What looks new here (to be checked by a fuller search)

1. **Ternary weights as the only per-weight training state.** DQT and ECO drop the master copy but keep per-weight
   optimizer state (Adam moments / momentum); Bop keeps a per-weight momentum; Q-GaLore keeps 8-bit weights. We keep 1.6
   bits per weight and $O(r(N+K))$ optimizer state per layer.
2. **Stochastic flips driven by a low-rank momentum**, with a sign gate and a factored second-moment step, at LLM scale
   (110M-340M so far), from scratch, 0.049 nats from master weights at 110M.
3. **The mechanism**: the gap to master sits in rare token pairs; its cause is the momentum's memory length; the
   optimum memory (~200-400 steps), induction heads forming once the memory is long enough.
4. **Dry friction on the whole momentum** as a decay that sets its own memory length (220 -> 390 steps), matching the best
   fixed decay.
5. **Rank findings**: rank matters only once the memory is long; the rank penalty stays roughly constant from 110M to
   340M; 8-bit storage of the momentum factors costs nothing.

## Baselines a reviewer will ask for

- **DQT (ternary, stochastic rounding + AdamW)** on our setup: same data family and sizes, the most direct comparison.
- **ECO-style error feedback** at ternary precision, and **Bop** (per-weight momentum + threshold) on the LM: both with
  per-weight state, to show what the low-rank state costs or saves.
- **Q-GaLore at matched training memory.**
- Memory and throughput measured for each, not only estimated.

Not yet searched: 1-bit / low-bit optimizer-state work (8-bit Adam, 1-bit Adam, MicroAdam, Adam-mini), BinaryConnect-era
work with stochastic binarization, sign-based optimizers (signSGD, Lion) and flip-based training of binary transformers.
