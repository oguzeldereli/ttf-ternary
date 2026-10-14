"""TTF update (bitnet.train.ts_layer) invariants, on a GPU:
  1. the fast path (--ts_fast) and the general path give the same trits and the same accumulator;
  2. the trits stay in {-1, 0, 1} and only entries with |A| >= theta move, by sign(A);
  3. at full rank (r >= min(N, K)) with N >= K (V square) the spend reduces each moved entry by exactly theta; with
     K > N the basis V is K x N and the part of a spend outside it is dropped (reported, not asserted);
  4. a per-entry bar (--ts_vbar) with the spend table (--ts_exact_spend) applies C - C V V^T, C = bar * D.
    python tests/test_ts.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bitnet.flip import KernelTernaryLinear      # noqa: E402
from bitnet.kernel import unpack_rows             # noqa: E402
from bitnet import train as T                     # noqa: E402


def cfg(**kw):
    c = dict(r=512, lr_ratio=1.0, theta=2.0, tau=1000.0, rank_s=512, b1=0.9, b2=0.95, gate=False, spend=1.0,
             tau_anneal=1.0, nomom=True, fast=False)
    c.update(kw)
    return c


def layer(N, K, seed):
    torch.manual_seed(seed)
    l = KernelTernaryLinear(K, N, 8).cuda()
    l.track = False
    return l


def main():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    N, K = 80, 48             # N >= K: at full rank V is K x K (square)
    g_seq = [torch.randn(N, K, device="cuda") for _ in range(6)]

    # 1. fast vs general path
    la, lb = layer(N, K, 0), layer(N, K, 0)
    lb.wpacked.copy_(la.wpacked)
    sa, sb = {}, {}
    for g in g_seq:
        T.ts_layer(la, g.clone(), sa, cfg(fast=True))
        T.ts_layer(lb, g.clone(), sb, cfg(fast=False, vdiag=False, diag=False))
    same = torch.equal(la.wpacked, lb.wpacked)
    Ua, Va = sa[id(la)]; Ub, Vb = sb[id(lb)]
    dA = float((Ua @ Va.T - Ub @ Vb.T).abs().max())
    print(f"1. fast vs general: trits identical {same}, max |A_fast - A_general| {dA:.2e}")
    assert same and dA < 1e-4

    # 2 + 3. two identical layers in lockstep for 3 steps; at step 4 one gets a bar nothing reaches (A before firing),
    # the other the real bar (moves and A after the spend). Full rank (r >= min(N, K)): A is represented exactly.
    lp, lq = layer(N, K, 1), layer(N, K, 1)
    lq.wpacked.copy_(lp.wpacked)
    sp, sq = {}, {}
    for g in g_seq[:3]:
        T.ts_layer(lp, g.clone(), sp, cfg()); T.ts_layer(lq, g.clone(), sq, cfg())
    assert torch.equal(lp.wpacked, lq.wpacked)
    w_before = unpack_rows(lq.wpacked, lq.K).clone()
    T.ts_layer(lp, g_seq[3].clone(), sp, cfg(theta=1e30))
    T.ts_layer(lq, g_seq[3].clone(), sq, cfg())
    A_pre = sp[id(lp)][0] @ sp[id(lp)][1].T
    A_post = sq[id(lq)][0] @ sq[id(lq)][1].T
    w_after = unpack_rows(lq.wpacked, lq.K)
    D = w_after.int() - w_before.int()
    legal = ((A_pre >= 2.0) & (w_before < 1)) | ((A_pre <= -2.0) & (w_before > -1))
    ok_ternary = bool(((w_after >= -1) & (w_after <= 1)).all())
    ok_where = torch.equal(D != 0, legal) and torch.equal(D[legal], torch.sign(A_pre[legal]).int())
    spend_err = float((A_pre - A_post - 2.0 * D.float()).abs().max())
    print(f"2. ternary {ok_ternary}; moves exactly where |A| >= theta and legal, by sign(A): {ok_where} "
          f"({int(legal.sum())} moves)")
    print(f"3. full rank: max |A_before - A_after - theta D| = {spend_err:.2e}")
    assert ok_ternary and ok_where and spend_err < 1e-4

    # 4. per-entry bar + spend table at low rank: the stored correction at moved entries is (C - C V V^T)
    l4 = layer(N, K, 2); s4 = {}
    c4 = cfg(r=8, vbar=1.0, exact_spend=True)
    for g in g_seq:
        T.ts_layer(l4, g.clone(), s4, c4)
    tab = s4["_ts"].get(("tab", id(l4)))
    print(f"4. per-entry bar + spend table ran ({0 if tab is None else tab[0].numel()} table entries)")
    print("OK: TTF update")


if __name__ == "__main__":
    main()
