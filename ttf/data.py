"""Training batches: random windows of the token stream, and the precomputed teacher top-k windows (--kd-cache)."""
from __future__ import annotations
import os
import time
import numpy as np
import torch


_KD_SHARD = {}


def _kd_window(args, idx):
    """--kd-cache: (start, ids [seq, k], log-probs [seq, k]) of window idx of the precomputed teacher targets (shards
    kd_00000.npz, ... of B windows each); waits for a shard that is still being written, wraps around once the cache
    is marked done."""
    import glob
    if "B" not in _KD_SHARD:
        while not glob.glob(os.path.join(args.kd_cache, "kd_00000.npz")):
            time.sleep(30)
        _KD_SHARD["B"] = len(np.load(os.path.join(args.kd_cache, "kd_00000.npz"))["starts"])
    B = _KD_SHARD["B"]
    sh, j = divmod(max(idx, 0), B)
    fn = os.path.join(args.kd_cache, f"kd_{sh:05d}.npz")
    if _KD_SHARD.get("sh") != sh:
        waited = 0
        while not os.path.exists(fn):
            if os.path.exists(os.path.join(args.kd_cache, "done")):     # the cache ran out (or has a gap: a shard
                have = sorted(glob.glob(os.path.join(args.kd_cache, "kd_*.npz")))   # claimed but never written):
                fn = have[sh % len(have)]; break                               # the existing shards, in order
            if waited % 600 == 0:
                print(f"--kd-cache: waiting for {fn}", flush=True)
            time.sleep(30); waited += 30
        z = np.load(fn)
        _KD_SHARD.update(sh=sh, starts=z["starts"], ids=z["ids"], lp=z["lp"], sids=z["sids"] if "sids" in z else None)
    j = j % len(_KD_SHARD["starts"])                                    # a shorter (last) shard: wrap within it
    if getattr(args, "kd_cache_mode", "") == "sample":
        assert _KD_SHARD["sids"] is not None, f"--kd-cache-mode sample: {fn} has no tail samples (teacher_topk.py --tail)"
        return int(_KD_SHARD["starts"][j]), *_kd_sampled(_KD_SHARD["ids"][j], _KD_SHARD["lp"][j], _KD_SHARD["sids"][j])
    return int(_KD_SHARD["starts"][j]), _KD_SHARD["ids"][j], _KD_SHARD["lp"][j]


def _kd_sampled(ids, lp, sids):
    """--kd-cache-mode sample: the top k (exact) plus the S tail samples, each with weight p_rest / S, as one id list
    and log-weights [seq, k + S] that sum to 1 per position: the KD target is then an unbiased estimate of the teacher's
    full distribution (duplicate samples add up in the loss's gather / scatter)"""
    pk = np.exp(lp.astype(np.float32))
    pr = np.clip(1.0 - pk.sum(-1, keepdims=True), 0.0, None)
    S = sids.shape[-1]
    lw = np.concatenate([np.log(np.maximum(pk, 1e-30)), np.broadcast_to(np.log(np.maximum(pr / S, 1e-30)), sids.shape)], -1)
    return np.concatenate([ids, sids], -1), lw.astype(np.float32)


def _kd_cache_batch(args, idx, data, seq_len, device, bs=1):
    """--kd-cache: micro-batch idx = windows idx * bs ... idx * bs + bs - 1 of the cache (bs 1: window idx, as before).
    Returns x, y [bs, seq] and (ids [bs * seq, k], log-probs [bs * seq, k]) on device, rows in the order of x.flatten."""
    ws, ids_, lps_ = [], [], []
    for b in range(bs):
        s0, ids, lp = _kd_window(args, idx * bs + b)
        ws.append(torch.from_numpy(data[s0:s0 + seq_len + 1].astype(np.int64)))
        ids_.append(torch.from_numpy(ids.astype(np.int64))); lps_.append(torch.from_numpy(lp))
    if str(device).startswith("cuda"):   # pinned + non-blocking: a copy from pageable memory waits for the GPU's queue
        put = lambda t_: t_.pin_memory().to(device, non_blocking=True)
    else:
        put = lambda t_: t_.to(device)
    w = put(torch.stack(ws))
    return w[:, :-1], w[:, 1:], (put(torch.cat(ids_)), put(torch.cat(lps_)))


def get_batch(data, bs, seq_len, device, gen=None):
    # gen: dedicated RNG so the data order is identical across runs regardless of
    # what else consumes the global RNG (different modes draw differently).
    ix = torch.randint(len(data) - seq_len - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
