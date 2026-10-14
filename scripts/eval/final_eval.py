"""Final evaluation on fixed windows, the same for every model and independent of the training seed.

    python scripts/eval/final_eval.py RUN_DIR [RUN_DIR ...] [--fw 2000] [--seed N]

RUN_DIR holds eval.pt (from scripts/eval/rescue.sh) or ckpt.pt / ckpt_9154.pt. Writes RUN_DIR/final_eval.json and
RUN_DIR/final_eval.npz:
  - wiki: every non-overlapping 2048-token window of data/wiki32k_val.bin (974 windows, 2.0M tokens);
  - fw:   the first --fw non-overlapping windows of data/fwedu32k_val.bin (FineWeb-Edu, another domain);
  per-window mean loss (for paired comparisons, scripts/eval/compare.py), per-position mean loss, overall mean and
  bits per byte (bytes per token measured on the val sets with the same tokenizer, EOT counted as a token:
  Wikipedia 3.569, FineWeb-Edu 4.003; the earlier docs used 3.58 for Wikipedia).
Check: the training-time evaluation (30 random batches of 16 seeded with the run's seed + 12345, see train.py
evaluate) is recomputed and compared with the FINAL val loss in train.log, so a wrongly rebuilt model shows up.

The model is rebuilt from the checkpoint alone: mode, config and layer options are saved in it, and the float extras
(row / column scales, additive low-rank adapter, attention temperature) are switched on when their parameters are in
the state dict; load_state_dict is strict.
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from bitnet.bitlinear import STATE                                   # noqa: E402
from bitnet.flip import build_kernel_transformer, enable_rc_scales, KernelTernaryLinear  # noqa: E402
from bitnet.master import build_master_transformer, MasterTernaryLinear  # noqa: E402
from bitnet.model import Attention                                   # noqa: E402

BYTES_PER_TOKEN_WIKI = 3.569
BYTES_PER_TOKEN_FW = 4.003


def load_model(path, device, mag_kind="add"):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)   # mmap: a 35 GB master checkpoint
    mc, mode, sd = ck["cfg"], ck["mode"], ck["model"]                         # need not fit in host memory
    if mode == "kernel":
        model = build_kernel_transformer(mc, grad_checkpoint=False, rate=0.0, beta=ck.get("beta", False),
                                         int8=ck.get("int8", False), dw_mode=ck.get("dw_mode", "int8"),
                                         int8_dx=ck.get("int8_dx", False), g_ref=ck.get("g_ref", 3.0))
    elif mode == "master":
        dt = next(v.dtype for k, v in sd.items() if k.endswith(".weight") and v.dim() == 2 and "tok_emb" not in k)
        model = build_master_transformer(mc, grad_checkpoint=False, dtype=dt)
    else:
        raise SystemExit(f"mode {mode} not supported")
    model = model.to(device)
    if "layers.0.ffn.router.weight" in sd:                   # --moe E:K:H (top-K from the run's arguments, else 2)
        from bitnet.model import enable_moe, BitTransformer
        E_ = sd["layers.0.ffn.router.weight"].shape[0]
        H_ = next(v.shape for k, v in sd.items() if k.startswith("layers.0.ffn.experts.0.w_gate.")
                  and k.split(".")[-1] in ("weight", "wpacked"))[0]
        a_ = ck.get("args") or {}
        a_ = a_ if isinstance(a_, dict) else vars(a_)
        k_ = int(str(a_.get("moe") or f"{E_}:2").split(":")[1])
        mk = (lambda i, o: MasterTernaryLinear(i, o, mc.act_bits, dtype=dt).to(device)) if mode == "master" else \
             (lambda i, o: KernelTernaryLinear(i, o, mc.act_bits, rate=0.0, beta=ck.get("beta", False),
                                               int8=ck.get("int8", False), dw_mode=ck.get("dw_mode", "int8"),
                                               int8_dx=ck.get("int8_dx", False), g_ref=ck.get("g_ref", 3.0)).to(device))
        enable_moe(model, f"{E_}:{k_}:{H_}", mk)
    if any(k.endswith(".bpacked") for k in sd):              # --ts_boost
        from bitnet.flip import enable_boost
        enable_boost(model)
    if "tok_emb.table.wpacked" in sd:                         # --tern_emb tied_wide:M
        from bitnet.tern_emb import enable_tern_emb
        M_ = sd["tok_emb.table.col_scale"].numel() // sd["tok_emb.proj.row_scale"].numel()
        enable_tern_emb(model, f"tied_wide:{M_}", "kernel")
    elif "tok_emb.wpacked" in sd or "tok_emb.latent" in sd:     # --tern_emb
        from bitnet.tern_emb import enable_tern_emb
        enable_tern_emb(model, "untied" if ("lm_head.wpacked" in sd or "lm_head.latent" in sd) else "tied",
                        "master" if mode == "master" else "kernel")
    if any(k.endswith("qk_logscale") for k in sd):
        for a in model.modules():
            if isinstance(a, Attention):
                a.qk_logscale = nn.Parameter(torch.zeros(a.n_heads, device=device))
    if any(k.endswith("row_scale") for k in sd):
        if mode == "master":
            for m in model.modules():
                if isinstance(m, MasterTernaryLinear):
                    m.enable_rc_scale()
        else:
            enable_rc_scales(model)
    loras = [k for k in sd if k.endswith("lora_B")]       # --m_lora R: low-rank latent baseline
    if loras:
        r = sd[loras[0]].shape[1]
        for m in model.modules():
            if isinstance(m, MasterTernaryLinear):
                m.enable_lora(r)
    mags = [k for k in sd if k.endswith("mag_A")]
    if mags:
        r = sd[mags[0]].shape[1]
        for m in model.modules():
            if isinstance(m, (KernelTernaryLinear, MasterTernaryLinear)):
                m.enable_lowrank_mag(mag_kind, r)
    tmags = [k for k in sd if k.endswith("mag_At.col_scale")]      # --lowrank_mag taddw:R (trits, rank 20 R)
    if tmags:
        r = sd[tmags[0]].shape[0] // 20
        for m in model.modules():
            if isinstance(m, KernelTernaryLinear):
                m.enable_lowrank_mag("taddw", r)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ck


@torch.no_grad()
def window_losses(model, data, L, n, device, bs=4):
    """mean loss of each of the first n non-overlapping windows (inputs data[i L : i L + L], targets shifted by 1)"""
    n = min(n, (len(data) - 1) // L)
    per_win = np.zeros(n); per_pos = torch.zeros(L, device=device)
    for s in range(0, n, bs):
        idx = range(s, min(s + bs, n))
        x = torch.stack([torch.from_numpy(data[i * L:i * L + L].astype(np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(data[i * L + 1:i * L + L + 1].astype(np.int64)) for i in idx]).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                               reduction="none").reshape(len(idx), L)
        per_win[s:s + len(idx)] = loss.mean(1).cpu().numpy()
        per_pos += loss.sum(0)
    return per_win, (per_pos / n).cpu().numpy()


@torch.no_grad()
def train_protocol_loss(model, data, seed, device, bs=16, L=2048, iters=30):
    """the training-time evaluate(): 30 batches of 16 random windows, generator seeded with seed + 12345"""
    gen = torch.Generator().manual_seed(seed + 12345)
    out = []
    for _ in range(iters):
        ix = torch.randint(len(data) - L - 1, (bs,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + L].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + L].astype(np.int64)) for i in ix]).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            _, loss = model(x, y)
        out.append(loss.item())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--wiki", default="data/wiki32k_val.bin")
    ap.add_argument("--fwval", default="data/fwedu32k_val.bin")
    ap.add_argument("--fw", type=int, default=2000, help="FineWeb-Edu windows (0: skip)")
    ap.add_argument("--seed", type=int, default=None, help="training seed (default: from the run name, else 1337)")
    ap.add_argument("--mag_kind", default="add")
    ap.add_argument("--no_check", action="store_true")
    a = ap.parse_args()
    device = "cuda"
    assert torch.cuda.is_available(), "GPU only (never the laptop GPU: run on a lab PC / 4090 / Myriad)"
    STATE.updates_enabled = False
    wiki = np.memmap(a.wiki, dtype=np.uint16, mode="r")
    fw = np.memmap(a.fwval, dtype=np.uint16, mode="r") if a.fw and os.path.exists(a.fwval) else None
    for run in a.runs:
        t0 = time.time()
        path = next(p for p in (os.path.join(run, f) for f in ("eval.pt", "ckpt.pt", "ckpt_9154.pt")) if os.path.exists(p))
        model, ck = load_model(path, device, a.mag_kind)
        L = 2048
        res = {"run": os.path.basename(os.path.normpath(run)), "ckpt": path, "step": ck.get("step"), "mode": ck["mode"]}
        w, wp = window_losses(model, wiki, L, 10 ** 9, device)
        res.update(wiki_loss=float(w.mean()), wiki_windows=len(w), wiki_bpb=float(w.mean() / math.log(2) / BYTES_PER_TOKEN_WIKI))
        arrays = {"wiki": w, "wiki_pos": wp}
        if fw is not None:
            f, fp = window_losses(model, fw, L, a.fw, device)
            res.update(fw_loss=float(f.mean()), fw_windows=len(f), fw_bpb=float(f.mean() / math.log(2) / BYTES_PER_TOKEN_FW))
            arrays.update(fw=f, fw_pos=fp)
        if not a.no_check:
            m = re.search(r"seed(\d+)", res["run"])
            seed = a.seed if a.seed is not None else (int(m.group(1)) if m else 1337)
            res["train_protocol_loss"] = train_protocol_loss(model, wiki, seed, device)
            log = os.path.join(run, "train.log")
            fin = re.findall(r"FINAL val loss ([\d.]+)", open(log).read()) if os.path.exists(log) else []
            res["logged_final"] = float(fin[-1]) if fin else None
            res["check_diff"] = None if not fin else res["train_protocol_loss"] - float(fin[-1])
            if res["check_diff"] is not None and abs(res["check_diff"]) > 1e-3:   # the model was not rebuilt as trained
                res["check_failed"] = True                                        # (e.g. --mag_kind mul run read as add)
                print(f"WARNING {res['run']}: rebuilt model's training-protocol loss {res['train_protocol_loss']:.4f} "
                      f"differs from the logged final {fin[-1]} by {res['check_diff']:+.4f}: its numbers are not valid",
                      flush=True)
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(os.path.join(run, "final_eval.json"), "w"), indent=1)
        np.savez(os.path.join(run, "final_eval.npz"), **arrays)
        print(json.dumps(res), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
