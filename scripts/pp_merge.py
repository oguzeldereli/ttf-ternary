"""Join the stage checkpoints of a --pipe run into one single-GPU checkpoint (what an unpiped run with the same flags
resumes from, or a --pipe run with another number of stages).
  python scripts/pp_merge.py OUT_DIR OUT_DIR_rank1 ... --out merged/ckpt.pt
Each stage file holds its own tensors (the model's, its float tail's optimizer state, its layers' TTF state, the
others' entries None / empty); the rest of the blob (config, step, flags) is the same in all and taken from stage 0."""
import argparse, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", help="the stage directories in stage order (or their ckpt.pt files)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
blobs = [torch.load(d if d.endswith(".pt") else os.path.join(d, "ckpt.pt"), map_location="cpu", weights_only=False)
         for d in a.dirs]
steps = {int(b["step"]) for b in blobs}
assert len(steps) == 1, f"stages at different steps {steps}"
out = dict(blobs[0]); out["pipe"] = None
model = {}
for b in blobs:
    dup = set(model) & set(b["model"])
    assert not dup, f"tensors in two stages: {sorted(dup)[:4]}"
    model.update(b["model"])
out["model"] = model
if "opt" in blobs[0]:
    st = {}           # a stage resumed from a full checkpoint keeps the other stages' optimizer state on meta: skip those
    meta = lambda d: any(torch.is_tensor(v) and v.is_meta for v in d.values())
    for b in blobs:
        st.update({k: v for k, v in b["opt"]["state"].items() if not meta(v)})
    out["opt"] = {"state": st, "param_groups": blobs[0]["opt"]["param_groups"]}
for key, empty in (("lowrank", lambda v: v is None), ("ts_state", lambda v: not v), ("tern_emb_state", lambda v: v is None)):
    if blobs[0].get(key) is None:
        continue
    out[key] = [next((b[key][i] for b in blobs if not empty(b[key][i])), blobs[0][key][i]) for i in range(len(blobs[0][key]))]
if blobs[0].get("ts_ranks") is not None and blobs[0].get("lowrank") is not None:   # each layer's rank from its owner
    out["ts_ranks"] = [next((b["ts_ranks"][i] for b in blobs if b["lowrank"][i] is not None), blobs[0]["ts_ranks"][i])
                       for i in range(len(blobs[0]["ts_ranks"]))]
out["touched"] = {k: v for b in blobs for k, v in (b.get("touched") or {}).items()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save(out, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
n_lr = sum(v is not None for v in out.get("lowrank") or [])
print(f"merged {len(blobs)} stages at step {steps.pop()}: {len(model)} tensors, "
      f"{len(out.get('opt', {}).get('state', {}))} optimizer states, {n_lr} layers' TTF state -> {a.out}")
