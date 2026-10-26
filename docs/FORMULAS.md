# Formulas

Section 0 is the update the current trainer runs (`python -m ttf.train`, its options and defaults). Sections 1-6 are
the history: every update rule used in this project, with its flag and result, the flags being those of the trainer up
to 17 Oct 2026 (tag `pre-consolidation-2026-10-17`; the current options of the kept ones:
[OPTIONS_MAP.md](OPTIONS_MAP.md)); results there are final validation loss at 300M tokens (110M model, from scratch)
unless noted. Run details: the results database (`python scripts/results.py show NAME`).

Notation, per ternary layer: trits $T\in\{-1,0,1\}^{N\times K}$; the batch gradient with respect to the trits
$g=\partial L/\partial T$ ($N\times K$, one batch of $16\times2048$ tokens); the momentum $M$; step $t$ of $T_{\max}=9155$.

## 0. The current trainer (`--mode ttf`)

Per ternary layer and step, with $\rho_t=(\text{lr}_t/\text{lr}_\text{peak})^{p}$ (`--ttf-step-p` $p$, 0.75):

- **Second moment**, factored (`--ttf-b2` 0.95, bias-corrected): $R_i\leftarrow\beta_2R_i+(1-\beta_2)\overline{g_{i\cdot}^2}$,
  $C_j$ likewise, $v_{ij}=R_iC_j/\overline R$. With `--dw factored` (the default) $g=g_y^\top x$ is never formed: the
  sums of $g^2$ come from a `--dw-sketch` $k$-column random sketch (256) and every product with $g$ from the factors.
- **Step**: $u=-\rho_t\,g/(\sqrt v+\epsilon)$ (no momentum).
- **Accumulator** $A\approx UV^\top$ at rank $r$ (`--ttf-rank`, 512), leak $\lambda=1-\rho_t/\tau$ (`--ttf-tau` 1000), one
  subspace iteration per step on the whole tally $\tilde A=\lambda A+u$:
  $$V'=\operatorname{orth}\big(\tilde A^\top U\big)=\operatorname{orth}\big(V(U^\top U)+u^\top U\big),\qquad U'=\tilde AV'=U(V^\top V')+uV'$$
  ($U$ here already leaked; `--ttf-orth` the orthonormalisation).
- **Firing** at $\theta$ (`--ttf-theta`, 16 from scratch; conversions use larger): $D=\operatorname{sign}(A)\odot
  \mathbb 1[|A|\ge\theta]$ where the trit stays in $\{-1,0,1\}$, $T\leftarrow T+D$; **spend**
  $U\leftarrow U-c\,\theta\,(DV)$ (`--ttf-spend` $c$, 1). $A$ is formed one tile at a time inside the firing kernel.
- **Micro-batches** (`--accum M` > 1): micro-batch $i$'s $u_i$ uses $\tilde v_i=\beta_2\hat v_{t-1}+(1-\beta_2)M^2\overline{g_i^2}$
  and enters on the step's basis, $U\leftarrow U+u_iV$ (the leak once, before the first). After the last one the basis
  turns once on the step's whole tally $\tilde A=\lambda A+\sum_iu_i$: $\tilde A^\top U=V(U^\top U)+(I-VV^\top)(\sum u)^\top U$
  and $\tilde AV'=UV^\top V'+(\sum u)(I-VV^\top)V'$, the two products with $\sum u$ read from the $k$-row and $k$-column
  sketches of $\sum g$ that $v$ keeps anyway (no dense sum). Firing and the spend once per step. (Before that the micro-batch turn
  used $M\,u_M^\top U$ from the last micro-batch alone.)
- **History of the turn**: from ttf-core 2858771 until 26 Oct the turn kept only the part of the step inside the old
  basis ($A\leftarrow A_{\text{in }V}V'V'^\top$), for one batch per step as well: at 110M from scratch 39% of the trits
  had never moved by step 2000 against 30% with the rule above, and the validation loss was 0.0085 worse (3.2483 vs
  3.2398 at 2000 steps), most of it in the MLP down projection (ever moved 25% vs 69%).
- **Options under evaluation** (off by default): `--ttf-band F` keeps the exact tally on the share F of entries nearest
  $\theta$ (one batch per step); `--ttf-hspend F` spends a move also from the row's input-correlated entries
  ($V_s=V+F(CV-V)$, $C$ the correlation of an EMA of $x^\top x$).
- **Float parts** (embedding, norm gains, row / column scales, adapters, attention temperature): AdamW in fp32
  (`--tail fp32`; `lean`: LeanAdam, the 27B recipes' lean optimizer state). `--intk-emb E:H` instead keeps the embedding / head as E- / H-bit codes in
  groups trained by the same rule (threshold `--ttf-theta` x `--intk-theta-mult`).
- **Ternary layer**: $y=\beta\,(r\odot(T\,(c\odot(x-\mu))))+\text{bias}$, $\beta=1/\sqrt{K\rho}$ from the trits'
  nonzero share $\rho$, 8-bit per-token activations, int8 tensor-core GEMMs (exact), the input gradient as two int8
  slices (`--dx int8x2`).

## 1. The model

**Ternary layer (kernel mode).** Weights stay packed at 5 trits per byte (1.6 bit). Forward:

$$y = \beta\,(r\odot(T\,x))\odot c \;+\; A\,(B^\top x), \qquad \beta = \frac{1}{\sqrt{K\rho}},\ \rho=\text{share of nonzero trits}$$

- $\beta$: variance-preserving scale computed from the trits themselves (no stored scale; the fix of 25 Sep).
- $r\in\mathbb R^N,\ c\in\mathbb R^K$: learned row and column scales (`--rc_scale`; $N+K$ floats, AdamW, no decay).
- $A\in\mathbb R^{N\times R},\ B\in\mathbb R^{K\times R}$: the additive low-rank float term ("sharp": `--lowrank_mag
  add:16`, $R=16$, AdamW with weight decay 0.1 `--mag_wd 0.1`). The multiplicative form `mul:R`,
  $W=\beta T\odot(1+AB^\top)$, was worse.
- Attention: per-head temperature $s_h$ (`--qk_temp`): logits $\times\, e^{s_h}$.
- Activations: 8-bit per-token absmax; int8 tensor-core forward (`--int8`).

**Float tail** (embeddings, norm gains, $r, c, A, B, s_h$): fp32 + AdamW (`--tail_fp32`), LR cosine
$1.5\cdot10^{-3}\to1.5\cdot10^{-4}$, 305 warmup steps.

**Master weights (the reference, `--mode master`).** A float latent $W$ per weight, BitNet b1.58 absmean quantization in
the forward, straight-through gradient, AdamW on $W$:

$$\gamma=\operatorname{mean}|W|,\qquad W_q=\gamma\cdot\operatorname{clip}(\operatorname{round}(W/\gamma),-1,1),\qquad \partial L/\partial W := \partial L/\partial W_q$$

16 bytes per weight in training (fp32 $W$ + two fp32 Adam moments + gradient). Result 2.751. Full precision
(`fp32_baseline`): 2.683. Master with its latent stored on a $(2^k-1)$-level grid (`--master_bits k`): 4 bits 2.923,
3 bits 3.122, 2 bits 3.469 (branches at 131M).

## 2a. The two-timescale rule (`--ts`, 2 Oct): the new best

Built on what master turned out to need (2b). Best: all full rank `ts16fullla1000_s0` **2.6746** (master 2.7513,
full precision 2.683); both ranks 512 `ts16rs512tau1000an05_s0` **2.7067**. Flags (rank 512 version):
`--ts --lowrank 512 --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5 --rc_scale --lowrank_mag add:16
--mag_wd 0.1 --qk_temp` (`--ts_tau_anneal` instead of `--ts_anneal 0.5` for the annealed-leak version).

Per layer and step ($\rho_t = \text{lr}_t/\text{lr}_\text{peak}$):

**Second moment**, factored as in 2.2 but with Adam's decay and bias correction ($\beta_2=0.95$):
$R_i\leftarrow\beta_2R_i+(1-\beta_2)\overline{g_{i\cdot}^2}$, $C_j$ likewise, $v_{ij}=R_iC_j/\overline R\,/(1-\beta_2^t)$.

**Short momentum** (direction), rank $r_s$ (`--ts_rank_s`), $\beta_1=0.9$ (~10 steps), kept by subspace iteration:
$m\leftarrow\beta_1m+(1-\beta_1)g$. Adam's normalized step: $u=-\dfrac{m/(1-\beta_1^t)}{\sqrt{v}+\epsilon}$.

**Long accumulator** (the latent's sub-threshold position), rank $r$ (`--lowrank`), leak $\tau$ (`--ts_tau`):

$$A\leftarrow\Big(1-\frac{\lambda_t}{\tau}\Big)A+\rho_t^{\,p}\,u,\qquad \lambda_t=\rho_t^{\,p}\ \text{(`--ts_tau_anneal`) or }1,\quad p=\text{`--ts\_anneal`}\ (1)$$

(the code scales the leak by the same factor as the steps, $\rho_t^p$: $\rho_t$ for the annealed-leak runs with the
default $p=1$, $\rho_t^{0.5}$ for `--ts_tau_anneal --ts_anneal 0.5`, e.g. the 1.3B run)

(one subspace-iteration step with $\beta=1$ after the leak, as 2.1).

**Fire at a threshold** (no flip rate): a trit moves by $d_{ij}=\operatorname{sign}(A_{ij})$ the step $|A_{ij}|\ge\theta$
(`--ts_theta`, 16 ~ half a bin of master's latent over the peak lr), unless blocked at $\pm1$; optionally only where this
batch's gradient agrees (`--lr_gate`: no effect here). **Spend**: $A\leftarrow A-c\,\theta D$, low-rank
$U\leftarrow U-c\,\theta\,(DV)$, $D$ the trit changes, $c=$ `--ts_spend` (1 = to the new trit's centre; 2 = stay at the
crossed boundary, worse).

State per layer: $(r+r_s)(N+K)$ for the two low-rank matrices plus $N+K$ for $v$; at 512 / 512 about twice the old
rule's. Results (110M, 300M tokens, vs master 2.7513):

| setting | final |
|---|---|
| full / full, tau 1000, annealed leak | **2.6746** |
| full / full, tau 1000 | 2.6982 |
| 512 / full, annealed leak | 2.6968 |
| 512 / 512, tau 1000, p 0.5 | **2.7067** |
| 512 / 512, tau 1000, annealed leak (+ gate) | 2.7115 (2.7116) |
| 512 / 512, tau 1000 (seeds 1 / 2) | 2.7328 / 2.7191 |
| 512 / 512, tau 3000 | 2.7270 |
| 512 / 512, tau 300 | 2.8054 |
| 512 / 256, 256 / 512 | 2.7815, 2.7825 |
| 256 / 256 | 2.8136 |
| 128 / 128, 64 / 64 | 2.9088, 3.0183 |
| 512 / 128 (default short rank), tau 300 / 1000 | 2.9415 / 2.8561 |
| theta 8 / 24 (512 / 512, tau 1000) | 2.7354 (annealed leak) / 2.8103 |
| no flips (theta $10^9$): float extras only | 3.1725 |

The float extras (row/column scales, rank-16 additive adapter, qk temperature) are in every row; master with the same
extras (`mx_extras_*`) is the fair reference (running).

## 2c. Later TTF flags (5-13 Oct)

Final recipe (110M): `--ts --ts_nomom --lowrank 512 --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75
--ts_tau_anneal --ts_fused --gemm int8mm --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp`. Results of each flag:
paper Appendix "Index of mechanisms and results" and docs/RUN_INDEX.md (tag pre-consolidation-2026-10-17).

| flag | what it does |
|---|---|
| `--ts_nomom` | no short momentum: $u=-g/\sqrt v$ (factored $v$) goes straight into the accumulator |
| `--ts_micro` (with `--grad_accum M`) | each micro-batch's $u_i=-g_i/\sqrt{\tilde v_i}$, $\tilde v_i=\beta_2\hat v_{t-1}+(1-\beta_2)M^2\overline{g_i^2}$, enters the accumulator; the batch's $v$ from the micro-batches' squares plus sketched cross terms; firing and spend once per step (paper App. micro) |
| `--ts_start S` | trits frozen until step S: no evidence kept, nothing fires; $v$ and the floats train |
| `--ts_vdiag` | logs $g^2/v$ (layer mean `dg_g2_over_v`, max `dgmax_g2_over_v`) and the $|u|$ tail. With `--ts_micro` the ratio is per micro-batch, so its normal level is $1/M$ |
| `--ts_uclip C`, `--ts_urms D` | clip $u$ per entry at $C\,\rho$ / scale a layer's $u$ so its rms is at most $D$ (Adafactor-style) |
| `--ts_move_cap F` | at most a share F of a layer's weights move per step (largest $|A|/\text{bar}$) |
| `--ts_vbar K` | cost-aware bar: $\theta\,\big(\sqrt{R_i}/\mathrm{med}\sqrt R\cdot\sqrt{C_j}/\mathrm{med}\sqrt C\big)^K$, clamped to $[\theta,4\theta]$ |
| `--ts_theta_noise K` | bar $=\max(\theta, K\cdot\overline{\mathrm{rms}(A)})$ with an EMA (0.99) of the layer's $\mathrm{rms}(A)$, every step (runs away from scratch) |
| `--ts_theta_noise_win N` | with `--ts_theta_noise K`: the bar is measured once: N active steps with evidence kept and nothing firing, then bar $=\max(\theta, K\,\mathrm{rms}(A))$, fixed |
| `--ts_boost S` | a second trit plane $B$: $W=T+S\beta B$; evidence past the bar at a trit already at $\pm1$ sets $B$, the first move back clears it |
| `--ts_resid --ts_resid_sparse F` | the largest share F of what the rank-$r$ projection drops is kept (index + bf16 value) and fed back the next step; `--ts_resid_steps K` keeps it K steps |
| `--ts_exact_spend` | the spend is applied exactly at moved entries (a sparse correction), not through the low-rank basis |
| `--tern_emb tied` / `untied` / `tied_wide:M` | ternary embedding (and head): $E=s_0\,r c\,T$; `tied_wide:M`: $E=T_{V\times Md}\,P^\top$ with a ternary $P_{d\times Md}$ ($M=20$ = fp32's bytes), both trained by TTF |
| `--lowrank_mag taddw:R` | the rank-R float adapter $AB^\top$ as ternary factors $A_t$ (N x 20R), $B_t$ (K x 20R) at fp32's bytes, trained by TTF; $A_t$'s row scales start at 0 |
| `--moe E:K:H` | every feed-forward is E SwiGLU experts of hidden H, top-K routing by a float router, load-balancing loss `--moe_aux`; each expert its own accumulator (master: experts with latents) |
| `WORLD_SIZE` / `RANK` (data parallel) | the gradient of each layer is all-reduced before the TTF update; ranks stay bit-identical |
| `--init_hf DIR --init_hf_method als / gptq` | convert a pretrained model: trits and scales by alternating least squares or by GPTQ onto the ternary grid (calibration `--init_calib`); `--init_stream` converts one decoder layer at a time from the safetensors (sequential GPTQ) |
| `--distill W` | KL to the full-precision teacher (its parameters cast to bf16) |
| `--eval_first_n N` | before training, the loss over the first N val windows in order (as `scripts/quant/qwen_check.py`) |
| `--lr_rewarm S:W:F` | re-warm a decayed run from step S over W steps to F times the peak |

## 2b. What master cannot do without (master from scratch with one ingredient changed, `--m_*`, `ttf/master_opt.py`)

| flag | change | final (master 2.7513) |
|---|---|---|
| `--m_factv` | second moment factored (row x column) | 2.7513 |
| `--m_rank 512` | first moment rank 512 (subspace iteration) | 2.7710 |
| `--m_gate` | update only where the batch gradient agrees with m | 2.7325 |
| `--m_leak 1000` | $W\leftarrow W-(W-t\gamma)/\tau$: the offset from the trit centre forgets, tau 1000 | 2.7298 |
| `--m_snap` | a latent whose trit changes is set to $t\gamma$ | **2.8225** |
| `--m_leak 300` | ... tau 300 | **2.8479** |
| `--m_beta1 0.997 --m_beta2 0.999` | long first moment (`--m_beta2 0.999` alone: 2.7685) | **3.2446** |
| `--m_lag 0.02` | the trits follow round(W / gamma) with prob. 0.02 per step | **3.5214** |
| `--m_gfix S`, `--m_clamp C` | frozen gamma (no effect in a 500-step branch) / clamp (collapses: it lowers gamma) | - |

## 2. The previous recipe (dry friction + spend)

Best so far: `gvsharp_dryspend_r1024_s0` **2.7998** (dry friction + spend, rank 1024 = full rank at 110M, where every
matrix has a smaller side of 768) and, with a compressed (sublinear) momentum, `gvsharp_dryspend_r512_s0` **2.8215**
(rank 512). Flags:
`--lowrank 512 --lr_beta 1 --dry_vec 0.0303 --spend 3 --lr_gate --lr_vnorm 0.99 --rc_scale --lowrank_mag add:16
--mag_wd 0.1 --qk_temp`.

Per layer and step, in order:

**2.1 Low-rank momentum** (`--lowrank r`, `--lr_beta b`). $M\approx UV^\top$, $U\in\mathbb R^{N\times r}$,
$V\in\mathbb R^{K\times r}$ orthonormal. The target is $M\leftarrow bM+g$, kept at rank $r$ by one subspace-iteration
step (two thin GEMMs and a QR, no SVD):

$$V'=\operatorname{qr}\!\big(b\,V(U^\top U)+g^\top U\big),\qquad U'=b\,U(V^\top V')+g\,V'$$

Start: $U_0$ random orthonormal, $V=\operatorname{qr}(g^\top U_0)$, $U=gV$. State $r(N+K)$ floats per layer. The
rank is at most $\min(N,K)$: at 110M ($\min=768$) `--lowrank 1024` is full rank. With the
current recipe $b=1$ (no decay); the memory comes from dry friction (2.5).

**2.2 Factored Adam step** (`--lr_vnorm 0.99`). Row and column EMAs of $g^2$ (Adafactor-style, $N+K$ floats):

$$R_i\leftarrow0.99R_i+0.01\,\overline{g_{i\cdot}^2},\qquad C_j\leftarrow0.99C_j+0.01\,\overline{g_{\cdot j}^2},\qquad S_{ij}=\frac{M_{ij}}{\sqrt{R_iC_j/\overline R}}$$

**2.3 Sign gate** (`--lr_gate`). Flip only where the momentum and this batch's gradient agree:

$$S_{ij}\leftarrow S_{ij}\cdot\mathbb 1[\operatorname{sign}S_{ij}=\operatorname{sign}g_{ij}]$$

**2.4 Flip.** With $m=\operatorname{mean}|S|$ taken before the gate, $g_{\text{ref}}=3$:

$$p_{ij}=\rho_t\cdot\min\!\Big(\frac{|S_{ij}|}{g_{\text{ref}}\,m},1\Big),\qquad T_{ij}\leftarrow\operatorname{clip}\!\big(T_{ij}-\operatorname{sign}S_{ij},-1,1\big)\ \text{with probability }p_{ij}$$

Flip rate: linear warmup to 0.02 over 30 steps, then cosine to 0:
$\rho_t=0.02\cdot\tfrac12\big(1+\cos\pi\tfrac{t-30}{T_{\max}-30}\big)$ (`--rate_peak 0.02 --rate_warmup 30
--rate_schedule cosine`). Because $p$ divides by the momentum's own mean size, a global rescale of $M$ changes no flip.

**2.5 Dry friction on the whole momentum** (`--dry_vec D`, with `--lr_beta 1`). After all layers, with all layers as
one vector ($\|M\|^2=\sum_\ell\|U_\ell\|^2$ since $V$ is orthonormal) and $G$ a slow EMA of the total gradient norm:

$$G\leftarrow0.99\,G+0.01\,\|g\|,\qquad f=\frac{\max(\|M\|-D\,G,\ 0)}{\|M\|},\qquad U_\ell\leftarrow f\,U_\ell\ \ \forall\ell$$

With the relative flip rule this is a global decay $b_{\text{eff}}=f=1-DG/\|M\|$ that sets its own memory:
measured 0.9954 / 0.9965 / 0.9960 / 0.9974 (memory 219 / 283 / 251 / 388 steps) at steps 1000 / 3000 / 6000 / 9154 of
`gvsharp_dry_s0` (`scripts/analysis/eff_beta.py` at tag `pre-consolidation-2026-10-17`). $D=0.0303$ (1/33).

**2.6 Spend** (`--spend c`). A flip consumes the push that caused it. With $\Delta=T_{\text{new}}-T_{\text{old}}$ (so
$\Delta_{ij}=-\operatorname{sign}S_{ij}$ where a trit moved) and $m_0=\operatorname{mean}|M|$ before the Adam step:

$$U\leftarrow U+c\,m_0\,(\Delta V)$$

$c=3$ removes $g_{\text{ref}}$ times the mean momentum at each flipped entry (projected onto the momentum's column space).

**Memory.** Per layer: $r(N+K)$ (momentum) + $N+K$ (second moments) + $N+K$ (row/column scales) + $R(N+K)$ (adapter,
with AdamW states) floats, plus a few scalars. Sublinear in the $NK$ weights.

## 3. Tried: mechanisms on the momentum rule (no look-ahead, from scratch unless "branch")

Base for each row is noted. "sharp base" = gate + Adam step + sharp (2.9875).

| mechanism | flag | formula | result |
|---|---|---|---|
| **decay** | `--lr_beta b` | $M\leftarrow bM+g$ | sharp base: 0.97 2.988, 0.99 2.911, **0.995 2.878**, 0.998 2.891, 1 3.047 |
| dry friction strength | `--dry_vec D` | 2.5 | 0.01 2.956, 0.02 2.897, **0.0303 2.881**, 0.05 2.888, 0.1 3.003 |
| rank | `--lowrank r` | 2.1 | sharp base: 256 2.988, 512 2.963; with dry friction: 256 2.881, 512 2.836, **1024 2.820**; dry + spend: 256 2.870, 512 2.822 |
| spend | `--spend c` | 2.6 | on beta 1 2.943 (faded late); on dry 2.881 -> 2.870; on beta 0.99 2.911 -> 2.903; on beta 0.995 2.878 -> 2.881 |
| sign gate | `--lr_gate` | 2.3 | row/col base 3.131 -> 3.119 / 3.114 (its effect is its lower flip count: rate control 3.121) |
| factored Adam step | `--lr_vnorm 0.99` | 2.2 | 3.131 -> 3.106; with the gate **3.077 / 3.070** (the two need each other) |
| row/column scales | `--rc_scale` | 1 | plain 3.158 -> 3.131 |
| slow gate | `--slow_gate 64 --slow_beta 0.999` | flip only where $\operatorname{sign}M=\operatorname{sign}M_{\text{slow}}$, $M_{\text{slow}}$ rank 64, decay 0.999 | sharp base 2.962; + beta 0.99 2.944; + beta 1 + spend 2.971 |
| dry friction per weight | `--dry_w D` | full $M$: $M_{ij}\leftarrow\operatorname{sign}M_{ij}\max(\lvert M_{ij}\rvert-D\,\overline{\lvert g\rvert},0)$, then back to rank $r$ | 2.941 |
| low-discrepancy dither | `--dither_ld` | flip iff $\operatorname{frac}(h_{ij}+t\varphi)<p_{ij}$, $h_{ij}$ fixed hash, $\varphi=0.618$ | 2.991 (no change) |
| undo | `--undo` | flip back last step's move $d$ where $d\ne0$, $d\,g>0$, $\operatorname{sign}M=d$ | inert with the gate (2% of flips); stopped |
| speed reference | `--speed_ref 0.995` | $m\leftarrow$ slow EMA of mean$\lvert M\rvert$ | 3.1315 (no change) |
| per-row speed reference | `--speed_row b` | $S_{i\cdot}\leftarrow S_{i\cdot}\,m/\text{EMA}(\overline{\lvert M_{i\cdot}\rvert})$ | swing test only |
| velocity rule with a speed limit | `--lr_beta 1 --mom_ncap C --pfun_tanh 10` | cap $\|M\|\le C\,\text{EMA}\|g\|$; $p=\rho_t\tanh(\lvert M\rvert/(g_{\text{ref}}v_0))$, $v_0=10\,\text{EMA}(\overline{\lvert g\rvert})$ | cap 3 3.217; cap 1 ~3.83 (stopped) |
| asymmetric gravity | `--grav_up D` | $M\leftarrow\text{where}(\operatorname{sign}M\operatorname{sign}g<0,\,DM,\,bM)+g$ | 3.501; with gate + Adam 3.497 |
| angle-scaled decay | `--lr_adapt` | $b_t=b\cos(g,M)$ | 10M screen only |
| several decays | `--multibeta 0.8,0.95,0.99` | per layer the momentum that best predicted $g_t$ proposes | 3.193 (branch) |
| accumulate then flip | `--accum_flip 33` | no flips for $K$ steps, then flip $\lvert M_{ij}\rvert>z\,\text{rms}(M)$, restart | 3.280 (branch) |
| adaptive flip rate | `--adapt_rate` | controller keeping $\cos(g_t,M_{t-1})\approx0.03$ | 3.160 (branch) |
| 1/4 flip rate | peak rate x1/4 | | 3.155 (branch), 3.193 (scratch) |
| online flip selector | `--select` | MLP keeps the proposals later gradients agree with | 3.366 (stopped) |
| target-point mechanisms | `--mech v1`, `--mech user` | per-weight target displacement $D\leftarrow b_D(D-\text{move})+(1-b_D)(-g/h)$ | 3.12 (branch), 3.51-3.58 (scratch, 131M) |
| evidence counter (not stateless) | `--evidence_bits 3` | $c\leftarrow\operatorname{clip}(c+\text{tick},\pm L)$, flip at $\lvert c\rvert=L$, reset | 3.097 (branch) |
| stuck mask | `--lr_mask_stuck` | drop $g$ where the trit is already at the bound it pushes to | worse (+0.1 at 12M) |
| subspace refresh | `--lr_refresh` | replace the weakest directions by the gradient's top directions outside the subspace | no gain (20M screen) |

## 4. Tried: look-ahead (the 24-27 Sep phase)

**Look-ahead filter** (`--lookahead k`, `--lookahead_xbatch`): propose flips $\Delta$, compute $g'$ at $T+\Delta$ (on a
fresh batch with `xbatch`), keep flip $i$ iff the midpoint slope still descends, $\Delta_i(g_i+g'_i)<0$; $k$ re-check
passes. Costs $k$ extra forward/backward passes. Best: look-ahead x2 + rank-256 momentum 3.127; + sharp 2.992. Dropped
27 Sep: batch 48 without look-ahead (the same text read) matched it, and every rule since is without it.

**Spend with look-ahead** (`--lr_spend c`): kept flips consume $c\,m$ of $M$, as 2.6. No gain in the 20M screen.

## 5. Tried: stateless flips from the gradient (the first phase, before momentum)

$$p_{ij}=\rho_t\cdot\min\!\Big(\frac{|g_{ij}|}{g_{\text{ref}}\,\overline{|g|}},1\Big)$$

| variant | flag | result (val perplexity, 300M) |
|---|---|---|
| constant rate | `--rate 0.02` | 135.8 |
| cosine-annealed rate | `--rate_schedule cosine` | 98.6 |
| + look-ahead x1 | `--lookahead 1` | 65.3 (at 202M) |
| per-row normalisation | `--norm row` | no gain (10M screen) |
| reversed ramp (small gradients first) | `--inv_prob` | no gain (10M screen) |
| hump-shaped probability | `--hump`: $p=\min\big(\tfrac{\lvert g\rvert}{g_{\text{ref}}}(1-(\lvert g\rvert/g_0)^\alpha),1\big)\rho$ | no gain (10M screen) |
| frozen absolute threshold | `--abs_scale` | 160.7 |
| flip lockout (1 bit/weight) | `--flip_lockout N` | 102.7 |
| spatial error feedback | `--err_feedback` | 156.0 |
| greedy rate line search | `--rate_search` | not pursued (5M smoke test) |

For comparison, the current recipe is at perplexity 16.8 ($e^{2.8215}$) and master at 15.7.

## 6. Measurements (analysis scripts)

| quantity | formula | script |
|---|---|---|
| loss by pair frequency | held-out loss at positions whose (previous, target) pair occurs $n$ times in the training set, bucketed by $n$ | `loss_by_freq.py` |
| loss by context position | held-out loss by position buckets 0-1 / 2-15 / 16-127 / 128-511 / 512+; copy gain = loss on a shuffled repeat minus on an exact repeat of random tokens | `loss_by_pos.py` |
| rank capture | signal energy $\langle G_1,G_2\rangle$ of a bucket's held-out gradient from two disjoint halves; captured $\langle PG_1,PG_2\rangle$, $P$ = projection on $\operatorname{span}U\times\operatorname{span}V$; and $\cos(M,G)$ | `rank_capture.py` |
| effective decay of dry friction | $b_{\text{eff}}=1-D\,\|g\|/\|M\|$, memory $1/(1-b_{\text{eff}})$ | `eff_beta.py` |
| swing test (41 steps from one checkpoint) | true gradient $T_t$ from 32 batches; lag-$k$ autocorrelation $\cos(T_t,T_{t+k})$; uphill share $=\Pr[\text{move}\cdot T_t>0]$; never-turning = weights whose momentum does not follow a reversal of $T$ | `wave.py`, `master_wave.py` |
| where the flips land | expected flips, uphill share and first-order $\Delta L=\sum p\cdot\text{move}\cdot T$ by row-steepness fifth | `flip_where.py` |
