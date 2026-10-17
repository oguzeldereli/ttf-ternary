# For scripts/tests/gate.sh (CORE_FLAGS_FN): the reference's flags with those the core no longer has removed. Each
# removed flag was a no-op at the value the gate's configurations pass (the flip rule's rate and threshold, the dense
# weight gradient, the look-ahead off, the short-momentum rank under --ts_nomom).
core_flags() {
  python3 - "$2" <<'PY'
import sys
drop1 = {"--rate", "--rate_peak", "--rate_warmup", "--rate_schedule", "--g_ref", "--dw_mode", "--lookahead", "--ts_rank_s"}
drop0 = set()
a = sys.argv[1].split(); out = []; i = 0
while i < len(a):
    if a[i] in drop1: i += 2; continue
    if a[i] in drop0: i += 1; continue
    out.append(a[i]); i += 1
print(" ".join(out))
PY
}
