"""q38_dump_cmp.py: compare two q38_bench --dump files (e.g. --update-tf32 against fp32, and fp32
against fp32 for the run-to-run floor): per tensor the relative difference of the float state, and for the packed
trits the share of trits that differ (5 trits per byte, base 3).

  python scripts/analysis/q38_dump_cmp.py A.pt B.pt"""
import sys
import torch


def trits(p):
    p = p.to(torch.int16).flatten()
    out = []
    for _ in range(5):
        out.append(p % 3); p = p // 3
    return torch.stack(out, -1)


def main():
    A, B = torch.load(sys.argv[1]), torch.load(sys.argv[2])
    tot_d = tot_n = 0
    for k in sorted(A):
        a, b = A[k], B[k]
        if k.endswith("wpacked"):
            ta, tb = trits(a), trits(b)
            d = (ta != tb).sum().item(); n = ta.numel()
            tot_d += d; tot_n += n
            print(f"{k:60s} trits differ {d} / {n} ({d / n:.2e})")
        elif a.is_floating_point():
            a, b = a.float(), b.float()
            r = ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
            print(f"{k:60s} rel diff {r:.2e}")
    print(f"all trits differ {tot_d} / {tot_n} ({tot_d / max(tot_n, 1):.2e})")


if __name__ == "__main__":
    main()
