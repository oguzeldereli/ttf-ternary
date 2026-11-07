"""q38_train.py end to end on a tiny random Qwen3.8-Flash-Next config (P, 5 Nov 2026): the float model is the teacher
(its top-k + sampled tail as the KD cache would hold them), the ternary student (rounding conversion) trains layer-major
for some steps. Checks: the loss falls, all three kinds of moves happen, the layer-major gradient equals plain autograd
on the whole model (one step's float gradients), and a save / load resumes to the same next step.
  python tests/checks/q38_train.py"""
import copy, os, sys, tempfile, types
import numpy as np
import torch
from ttf.q38 import model as M
from ttf.q38 import train as QT
from ttf.data import _kd_sampled

dev = "cuda" if torch.cuda.is_available() else "cpu"
t = dict(hidden_size=64, hc_count=4, hc_lowrank=8, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
         head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
         num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
         vocab_size=1000, ngram_vocab_size_base=101, heads_per_ngram=2, ple_embed_dim=64, eos_token_id=999,
         bos_token_id=999, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=64,
         layer_types=["linear_attention"] * 3 + ["full_attention"], ple_layer_ids=[2], max_position_embeddings=4096,
         ngram_size=3, ple_conv_kernel_size=3, hidden_act="silu", rms_norm_eps=1e-6)
cfg = M.text_config({"text_config": t})
torch.manual_seed(0)
teacher = M.build(cfg, device="cpu")
with torch.no_grad():
    for n, p in teacher.named_parameters():
        if n.endswith("A_log"):
            p.copy_(torch.randn_like(p) * 0.1)
        elif p.dim() >= 2:
            p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
        else:
            p.copy_(torch.randn_like(p) * 0.1)
    for L in teacher.layers:
        if L.ple is not None:
            ng = L.ple.ple_embedding
            ng.table = torch.randn(ng.rows, ng.dim)
    teacher.lm_head.weight.mul_(12.0)           # (a peaked teacher: random weights alone give a near-uniform softmax)
dt = torch.bfloat16 if dev == "cuda" else torch.float32
teacher = teacher.to(dev, dt)

T, MB, K_, S_ = 48, 4, 16, 16
gen = torch.Generator().manual_seed(1)


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(MB):
        w = torch.randint(0, 999, (T + 1,), generator=g)
        with torch.no_grad():
            lp = torch.log_softmax(teacher(w[None, :-1].to(dev)).float()[0], -1).cpu()
        top = lp.topk(K_, -1)
        rest = lp.exp().scatter(1, top.indices, 0.0)
        sids = torch.multinomial(rest / rest.sum(-1, keepdim=True), S_, replacement=True, generator=g)
        ids, lw = _kd_sampled(top.indices.numpy(), top.values.numpy(), sids.numpy())
        out.append({"ids": w[None, :-1], "tgt": w[1:], "kd_ids": torch.from_numpy(ids), "kd_lw": torch.from_numpy(lw)})
    return out


student = copy.deepcopy(teacher).float().cpu()
for L in student.layers:
    QT.swap_ternary(L, adapter_rank=4, quick=True)
    for n, tm in QT.tern_modules(L):            # a nonzero adapter start (S's conversion fits one)
        tm.ta_packed.copy_(QT.pack3(torch.randint(-1, 2, (tm.E, tm.N, tm.ra), dtype=torch.int8)))
student = student.to(dt)
for L in student.layers:                        # (the ternary modules' floats stay fp32)
    for n, tm in QT.tern_modules(L):
        tm.float()
student.embed_tokens.to(dev); student.lm_head.to(dev); student.hyper_connection_mixer.to(dev); student.rotary.to(dev)
args = types.SimpleNamespace(steps=40, theta=2.0, tau=200.0, sketch=16, rank_ratio=0.5, lr=3e-3, lr_min=3e-4, warmup=2,
                             kd_weight=1.0, head_chunk=16, expert_chunk=4, pin=False, save_every=10 ** 9,
                             eval_every=10 ** 9)

# 1) the layer-major float gradients equal autograd on the whole model (no ternary update: theta huge, lr 0)
chk = copy.deepcopy(student)
tr = QT.Trainer(chk, types.SimpleNamespace(**{**vars(args), "theta": 1e30, "lr": 0.0, "lr_min": 0.0}), dev)
grads = {}
orig = tr.opt.step
def grab(key, named, lr):
    for n, p in named:
        if p.grad is not None:
            grads[(key, n)] = p.grad.detach().float().cpu().clone()
    orig(key, named, lr)
tr.opt.step = grab
b0 = batch(100)
tr.train_step(b0)
ref = copy.deepcopy(student).to(dev)
for L in ref.layers:
    QT.layer_floats(L)
nrm = sum(int((b["tgt"] != -1).sum()) for b in b0)
for b in b0:
    x = ref.head_in(_h := None) if False else None
    h = ref.embed(b["ids"].to(dev)); pe = ref.rope(T, dev)
    for L in ref.layers:
        h = L(h, pe, b["ids"].to(dev))
    xh = ref.head_in(h)[0]
    from ttf.qwen import _kd_topk_chunk
    l = _kd_topk_chunk(xh, ref.lm_head.weight, b["tgt"].to(dev), b["kd_ids"].to(dev), b["kd_lw"].to(dev), 1.0, mode="renorm")
    (l / nrm).backward()
worst = 0.0
for i, L in enumerate(ref.layers):
    for n, p in L.named_parameters():
        if p.grad is None or (i, n) not in grads:
            continue
        g0, g1 = grads[(i, n)], p.grad.float().cpu()
        rel = float((g0 - g1).norm() / g1.norm().clamp_min(1e-12))
        worst = max(worst, rel)
print(f"layer-major vs autograd: {len(grads)} float grads, worst relative diff {worst:.2e}")
assert len(grads) > 20 and worst < (3e-2 if dt == torch.bfloat16 else 1e-4), worst

# 2) training: loss falls, moves of every kind
tr = QT.Trainer(student, args, dev)
val = batch(999)
v0 = tr.eval_loss(val)
mv = {"exp": 0, "dense": 0, "adapter": 0}
for s in range(25):
    r = tr.train_step(batch(s))
    for k in mv:
        mv[k] += r["moves"][k]
    if s % 5 == 0:
        print(r)
v1 = tr.eval_loss(val)
print(f"val {v0:.4f} -> {v1:.4f}; moves {mv}")
assert v1 < v0 - 0.02, (v0, v1)
assert all(v > 0 for v in mv.values()), mv

# 3) save / load: the next step is the same
d = tempfile.mkdtemp()
tr.save(d)
ra = tr.train_step(batch(500))
st2 = copy.deepcopy(student)
tr2 = QT.Trainer(st2, args, dev)
tr2.load(d)
tr2.opt = tr2.opt                     # (loaded)
rb = tr2.train_step(batch(500))
print("resume", ra["loss"], rb["loss"], ra["moves"], rb["moves"])
assert ra["step"] == rb["step"] and ra["loss"] == rb["loss"] and ra["moves"] == rb["moves"], (ra, rb)
print("q38_train tiny end-to-end OK")

# 4) SIGINT mid-run (S's budget watchdog): the step finishes, a slim save to OUT/stop, run() returns
import json as _json, signal as _signal
out = tempfile.mkdtemp()
a4 = types.SimpleNamespace(**{**vars(args), "out": out, "steps": 10, "eval_every": 10 ** 9, "save_every": 10 ** 9,
                              "stop_save": "slim"})
st4 = copy.deepcopy(student)
tr4 = QT.Trainer(st4, a4, dev)
def _b(s):
    if s == 3:
        os.kill(os.getpid(), _signal.SIGINT)
    return batch(s)
QT.run(tr4, a4, _b, lambda: val)
meta = _json.load(open(os.path.join(out, "stop", "meta.json")))
assert meta == {"step": 4, "slim": True}, meta
blob = torch.load(os.path.join(out, "stop", "L0.pt"), weights_only=False)
assert set(blob) == {"layer"}
st5 = copy.deepcopy(student); st5.layers[0].load_state_dict(blob["layer"])
try:
    QT.Trainer(copy.deepcopy(student), a4, dev).load(os.path.join(out, "stop")); raise SystemExit("slim save resumed")
except AssertionError:
    pass
_signal.signal(_signal.SIGINT, _signal.default_int_handler)
print("stop on SIGINT OK:", meta)

# 5) --prefetch (the TTF state on a copy stream) gives the same steps as the plain path
if dev == "cuda":
    a5 = types.SimpleNamespace(**{**vars(args), "prefetch": True, "pin": True})
    a6 = types.SimpleNamespace(**{**vars(args), "prefetch": False, "pin": True})
    s5, s6 = copy.deepcopy(student), copy.deepcopy(student)
    t5, t6 = QT.Trainer(s5, a5, dev), QT.Trainer(s6, a6, dev)
    for s_ in range(3):
        r5, r6 = t5.train_step(batch(700 + s_)), t6.train_step(batch(700 + s_))
        assert r5["loss"] == r6["loss"] and r5["moves"] == r6["moves"], (r5, r6)
    for k in t5.ts:
        for kk in ("U", "V", "R", "C", "t"):
            assert torch.equal(t5.ts[k][kk], t6.ts[k][kk]), (k, kk)
    print("prefetch == plain over 3 steps:", r5["moves"])

# 6) --resident (layers stay on the device) + --wbatch 2 (windows batched per layer call) track the plain path (the
# batched calls sum in another order: close, not equal)
if dev == "cuda":
    a7 = types.SimpleNamespace(**{**vars(args), "resident": True, "wbatch": 2})
    a8 = types.SimpleNamespace(**{**vars(args)})
    s7, s8 = copy.deepcopy(student), copy.deepcopy(student)
    t7, t8 = QT.Trainer(s7, a7, dev), QT.Trainer(s8, a8, dev)
    for s_ in range(3):
        r7, r8 = t7.train_step(batch(800 + s_)), t8.train_step(batch(800 + s_))
        print("resident+wbatch2", r7["loss"], r7["moves"], "plain", r8["loss"], r8["moves"])
        assert abs(r7["loss"] - r8["loss"]) <= 2e-2 * abs(r8["loss"]), (r7, r8)
    print("resident + wbatch 2 ~ plain over 3 steps")
