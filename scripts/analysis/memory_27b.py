"""Memory of a ~23B-parameter ternary model (width 5120, 83 layers, MLP 13824, 8 KV heads of 128) trained with the
two-timescale rule, for different ranks and storage formats of the state. Estimate: weights, float tail (16-bit values,
gradients and 8-bit Adam, ~1.6 GiB), the gradient of the largest layer (update inside the backward) and the state;
activations are extra (~3 GiB at micro-batch 1 x 2048 with full checkpointing).
The second table is the budget on a 16 GB card anchored on what was measured (b27 smoke on barbel, 3 Oct: ranks 256,
int8 / fp16 state on the host, fp32 float tail + AdamW: torch peak 14.10 GiB, persistent 8.61 GiB, host RSS 13.7 GiB),
with each saving marked measured (at 110M-1.3B) or pending (a quality run still going).
    python scripts/analysis/memory_27b.py"""
GiB = 2**30
d, L, f, kv = 5120, 83, 13824, 1024
mats = [(d, d), (kv, d), (kv, d), (d, d), (f, d), (f, d), (d, f)]  # (out, in): q k v o gate up down
tern = L * sum(n * k for n, k in mats); P = tern + 0.29e9
nk = L * sum(n + k for n, k in mats); Nsum = L * sum(n for n, _ in mats); Ksum = L * sum(k for _, k in mats)
packed = tern / 5; tail = 1.6 * GiB; grad = max(n * k for n, k in mats) * 4


def row(name, r_s, r, bm, bU, bV, host=False):
    st = nk * r_s * bm + (Nsum * bU + Ksum * bV) * r + nk * 4  # momentum, accumulator U / V, factored second moment
    tot = packed + tail + grad + st
    gpu = tot - st + (2 * st / L if host else st)                # host: two layers' state on the GPU at a time
    print(f"{name:46s} state {st / GiB:5.2f} GiB | GPU {gpu / GiB:5.2f} GiB ({gpu / 1e9:4.1f} GB) + activations"
          f" | {tot / P:.2f} B/param ({16 / (tot / P):4.1f}x less than 16)")


print(f"ternary {tern / 1e9:.2f}B, all {P / 1e9:.2f}B params; packed trits {packed / GiB:.2f} GiB; largest gradient {grad / GiB:.2f} GiB")
row("all 8-bit, ranks 1024 (earlier estimate)", 1024, 1024, 1, 1, 1)
row("8-bit direction + 16-bit accumulator, 1024", 1024, 1024, 1, 2, 2)
row("  same, accumulator V in 8-bit (untested)", 1024, 1024, 1, 2, 1)
row("8-bit / 16-bit, ranks 512", 512, 512, 1, 2, 2)
row("8-bit / 16-bit, ranks 256", 256, 256, 1, 2, 2)
row("8-bit / 16-bit, ranks 1024, state in host RAM", 1024, 1024, 1, 2, 2, host=True)
row("fp32 state, ranks 1024", 1024, 1024, 4, 4, 4)


# ---- budget on one 16 GB card, anchored on the b27 measurement ----
TAIL_P = 0.29e9                       # tied embedding 0.164B + rank-16 adapters 0.119B + scales / gains
M_PERSIST, M_PEAK, M_HOST = 8.61, 14.10, 13.7
transient = M_PEAK - M_PERSIST        # checkpointed activations, recompute, one layer's update temporaries and state copies
fp32_tail = M_PERSIST - packed / GiB  # tail values + gradients + AdamW, fp32 (16 B / param)
lean_tail = TAIL_P * 5 / GiB          # bf16 / fp16 values + 16-bit gradients + int8 first moment (factored second moment)
rows_ = dict(zip("q k v o gate up down".split(), mats))


def state_gib(r, r_s, bm=1, ba=2, rs_type=None, bv=None):
    """momentum (rank r_s, or per type) bm bytes / entry, accumulator rank r with U at ba and V at bv (default ba) bytes,
    factored v fp32"""
    mom = sum(L * (n + k) * (rs_type[t] if rs_type else r_s) for t, (n, k) in rows_.items()) * bm
    return (mom + (Nsum * ba + Ksum * (ba if bv is None else bv)) * r + nk * 4) / GiB


host_base = M_HOST - state_gib(256, 256)  # host RSS besides the state (libraries, CUDA context, pinned staging)
print(f"\nbudget (16 GB card, micro-batch 1 x 2048, state on the host): packed {packed / GiB:.2f}, fp32 tail {fp32_tail:.2f}"
      f" (lean {lean_tail:.2f}), transient {transient:.2f}, host besides state {host_base:.2f} GiB")
print("| ranks (acc / momentum) | state format | evidence for quality | state GiB | GPU peak GiB (fp32 / lean tail) | host GiB |")
print("|---|---|---|---|---|---|")
half = lambda r: dict(q=r // 4, k=r // 4, v=r // 4, o=r // 4, gate=3 * r // 4, up=3 * r // 4, down=3 * r // 4)
for r, r_s, bm, ba, rst, fmt, ev, *bv in [
        (256, 256, 1, 2, None, "int8 / fp16", "measured run (b27 smoke); quality: +0.075 at 110M vs 512"),
        (512, 512, 1, 2, None, "int8 / fp16", "free at 110M (2 seeds)"),
        (512, 0, 1, 2, half(512), "int8 / fp16, momentum by type", "**costs** +0.04 (110M, to 4750)"),
        (512, 256, 1, 2, None, "int8 / fp16, momentum 256", "**costs** +0.034 (110M, at 2250)"),
        (512, 512, 0.5, 2, None, "int4 / fp16", "**fails** (+0.50 at 1000)"),
        (512, 512, 1, 2, None, "int8 / fp16, accumulator V int8", "costs +0.011 (110M final)", 1),
        (512, 512, 1, 1.5, None, "int8 / 12-bit", "5 mantissa bits free to 6750 (+0.0002); 12-bit packing not built"),
        (1024, 1024, 1, 2, None, "int8 / fp16", "110M full rank 2.6746 (other leak) vs 512 2.7067; 1.3B untested"),
        (1024, 1024, 1, 1.5, None, "int8 / 12-bit", "as above"),
        (1024, 512, 1, 1.5, None, "int8 / 12-bit, momentum 512", "momentum below the accumulator rank costs at 110M"),
]:
    st = state_gib(r, r_s, bm, ba, rst, bv[0] if bv else None)
    cp = 2 * st / L                                    # two layers' state copied in at a time
    gpu = packed / GiB + transient + cp - 2 * state_gib(256, 256) / L
    print(f"| {r} / {'by type' if rst else r_s} | {fmt} | {ev} | {st:.2f} | {gpu + fp32_tail:.1f} / {gpu + lean_tail:.1f} |"
          f" {st + host_base:.1f} |")
