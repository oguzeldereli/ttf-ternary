"""In-training probes of what the model has learned (read-only, a few forwards).

  logit_std        std of the logits over the vocab, averaged over positions
  bias_std         std over the vocab of the position-averaged logits: the size of the
                   context-independent (unigram) part of the prediction
  norm_gain        mean |gain| of the final RMSNorm;  emb_norm  mean embedding row norm
  loss_ctxK        loss on the next token given only the last K tokens (K=1 is the
                   bigram regime: previous token only)
  copy_first/second  loss on a random token sequence and on its exact repeat;
                   copy_gain = first - second (in-context copying, induction)
"""
import numpy as np
import torch
import torch.nn.functional as F

_CACHE = {}


def _batches(val, device, vocab):
    if "b" in _CACHE:
        return _CACHE["b"]
    g = np.random.default_rng(4321)
    n = len(val) - 600
    starts = g.integers(0, n, size=4)
    ctx = torch.from_numpy(np.stack([val[s:s + 513].astype(np.int64) for s in starts])).to(device)
    # windows ending at the same 1024 target positions, for context lengths K
    ends = g.integers(600, len(val) - 2, size=1024)
    wins = {K: torch.from_numpy(np.stack([val[e - K:e + 1].astype(np.int64) for e in ends])).to(device)
            for K in (1, 8, 64, 512)}
    rnd = torch.from_numpy(g.integers(1000, vocab - 1000, size=(16, 64))).to(device)
    copy = torch.cat([rnd, rnd], dim=1)
    _CACHE["b"] = (ctx, wins, copy)
    return _CACHE["b"]


def _last_loss(model, w, chunk=256):
    tot = 0.0
    for i in range(0, w.shape[0], chunk):
        x, y = w[i:i + chunk, :-1], w[i:i + chunk, 1:].clone()
        y[:, :-1] = -1                              # only the last position counts
        tot += model(x, y)[1].item() * x.shape[0]
    return tot / w.shape[0]


@torch.no_grad()
def probe(model, val, device):
    vocab = (model.lm_head.weight if model.lm_head is not None else model.tok_emb.weight).shape[0]
    ctx, wins, copy = _batches(val, device, vocab)
    was = model.training
    model.eval()
    out = {}
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        logits = model(ctx[:, :-1])[0].float()               # [4, 512, V]
        out["logit_std"] = logits.std(-1).mean().item()
        out["bias_std"] = logits.mean((0, 1)).std().item()
        out["pred_entropy"] = -(logits.log_softmax(-1).exp() * logits.log_softmax(-1)).sum(-1).mean().item()
        del logits
        for K, w in wins.items():
            out[f"loss_ctx{K}"] = _last_loss(model, w)
        x, y = copy[:, :-1], copy[:, 1:]
        first, second = y.clone(), y.clone()
        first[:, 63:] = -1                                   # targets inside the first copy
        second[:, :64] = -1                                  # targets inside the repeat
        second[:, 64] = -1                                   # (first repeated token: no copy cue yet)
        out["copy_first"] = model(x, first)[1].item()
        out["copy_second"] = model(x, second)[1].item()
        out["copy_gain"] = out["copy_first"] - out["copy_second"]
    out["norm_gain"] = model.norm.weight.float().abs().mean().item()
    out["emb_norm"] = model.tok_emb.weight.float().norm(dim=1).mean().item()
    if was:
        model.train()
    return out


def parse_schedule(spec):
    """'0-40:5,40-160:20' -> {0,5,...,35,40,60,...,140}"""
    steps = set()
    for part in filter(None, (spec or "").split(",")):
        rng, every = part.split(":")
        a, b = (int(v) for v in rng.split("-"))
        steps.update(range(a, b, int(every)))
    return steps
