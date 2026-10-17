"""Rank of the change in the trits between two snapshots.

TTF stores its not-yet-spent evidence at rank r; this asks whether the trits it
actually moved are low rank too. For each ternary layer, D = T_b - T_a (entries in
{-2..2}); report the share of ||D||^2 in the top r singular values, the
participation-ratio rank (sum s^2)^2 / sum s^4, and the same two numbers for a
control with D's entries shuffled within the matrix (same count, no structure).

  python scripts/analysis/dT_svd.py A.pt B.pt --rank 128 [--out rows.json]
"""
import argparse
import json
import re

import torch

from ttf.kernel import unpack_rows


def trits(sd):
    return {k[:-len(".wpacked")]: unpack_rows(v, v.shape[1] * 5)
            for k, v in sd.items() if k.endswith(".wpacked")}


def stats(D, r):
    s2 = torch.linalg.svdvals(D).pow(2)
    tot = s2.sum().item()
    if tot == 0:
        return float("nan"), float("nan")
    return s2[:r].sum().item() / tot, tot ** 2 / s2.pow(2).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--rank", type=int, nargs="+", default=[128])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A = trits(torch.load(args.a, map_location="cpu", weights_only=False)["model"])
    B = trits(torch.load(args.b, map_location="cpu", weights_only=False)["model"])
    g = torch.Generator().manual_seed(0)
    rows = []
    for name in A:
        if "tok_emb" in name or "lm_head" in name:
            continue
        D = (B[name].to(torch.int16) - A[name].to(torch.int16)).float()
        nz = int((D != 0).sum())
        Ds = D.flatten()[torch.randperm(D.numel(), generator=g)].view_as(D)
        row = {"layer": name, "shape": list(D.shape), "changed": nz / D.numel()}
        for r in args.rank:
            e, pr = stats(D, r)
            es, prs = stats(Ds, r)
            row[f"top{r}"], row[f"top{r}_shuf"] = e, es
            row["pr"], row["pr_shuf"] = pr, prs
        rows.append(row)
    types = {}
    for row in rows:
        t = re.sub(r".*\.", "", row["layer"])
        types.setdefault(t, []).append(row)
    keys = [k for k in rows[0] if k not in ("layer", "shape")]
    print("type     n " + " ".join(f"{k:>11s}" for k in keys))
    for t, rs in list(types.items()) + [("all", rows)]:
        print(f"{t:8s} {len(rs):2d} " + " ".join(f"{torch.tensor([x[k] for x in rs]).nanmean().item():11.4f}" for k in keys))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
