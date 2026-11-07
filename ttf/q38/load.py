"""A trained Q38 student from a slim checkpoint, with no Trainer (no TTF state): the model is built as in
ttf.q38.train.main (base float weights from the slim base checkpoint, the layers swapped to ternary), then each layer's
state comes from the slim checkpoint (SLIM/L{i}.pt, blob["layer"]). Layers live on the host except the first `resident`;
embed, head, stream mixer and rotary go to `dev`. For evaluation and analysis scripts (scripts/eval, scripts/analysis)."""
import json
import os

import torch

from ttf.q38 import model as QM
from ttf.q38.train import swap_ternary


def load_slim(model, slim, dev, layers=0, resident=0, log=print):
    """model: the slim base checkpoint (HF dir); slim: the run's slim save (meta.json + L{i}.pt); layers > 0: only the
    first N layers (smoke tests)"""
    cfg = QM.text_config(model)
    m = QM.build(cfg, layers=layers or None, device="meta")
    rd = QM.Reader(model)
    QM.load_globals(m, rd, "cpu")
    meta = json.load(open(os.path.join(slim, "meta.json")))
    for i, L in enumerate(m.layers):
        blob = torch.load(os.path.join(slim, f"L{i}.pt"), map_location="cpu", weights_only=False)["layer"]
        ra = next((int(v.shape[1]) for k, v in blob.items() if k.endswith("ad_cb")), 0)
        QM.load_layer(L, i, rd, "cpu", experts=False)
        swap_ternary(L, adapter_rank=ra)
        L.to("cpu")
        miss, unexp = L.load_state_dict(blob, strict=False)
        assert not unexp, f"L{i} unexpected {unexp[:5]}"
        miss = [k for k in miss if not k.startswith("ple.ple_embedding.table")]
        assert not miss, f"L{i} missing {miss[:5]}"
        if L.ple is not None:
            L.ple.ple_embedding.table, L.ple.ple_embedding.scale = QM.ngram_table_memmap(model, i)
        if i < resident:
            L.to(dev)
        del blob
        log(f"L{i} loaded (adapter rank {ra}){' resident' if i < resident else ''}")
    for mod in (m.embed_tokens, m.lm_head, m.hyper_connection_mixer):
        mod.to(dev)
    m.rotary.to(dev)
    m.eval()
    log(f"slim checkpoint {slim}: step {meta['step']}")
    return m
