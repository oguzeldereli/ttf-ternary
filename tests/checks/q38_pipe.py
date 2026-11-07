"""q38_pipe.py == q38_train.Trainer on a tiny random config (P, 5 Nov 2026): 2 stages (gloo, CPU, fp32) against one
Trainer, 3 steps + an eval: the same losses, moves and TTF state; pipeline groups of 3 over 4 micro-batches (an uneven
last group).
  python tests/checks/q38_pipe.py        (CPU only; 2 processes, 2 threads each)"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy, sys, tempfile, types
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ttf.q38 import model as M
from ttf.q38 import train as QT
from ttf.q38 import pipe as QP
from ttf.data import _kd_sampled

T, MB, K_, S_, STEPS = 48, 4, 16, 16, 3


def setup():
    torch.set_num_threads(2)
    t = dict(hidden_size=64, hc_count=4, hc_lowrank=8, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
             head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
             linear_value_head_dim=16, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
             shared_expert_intermediate_size=16, vocab_size=1000, ngram_vocab_size_base=101, heads_per_ngram=2,
             ple_embed_dim=64, eos_token_id=999, bos_token_id=999, indexer_n_heads=2, indexer_kv_heads=1,
             indexer_head_dim=16, indexer_budget=64, layer_types=["linear_attention"] * 3 + ["full_attention"],
             ple_layer_ids=[2], max_position_embeddings=4096, ngram_size=3, ple_conv_kernel_size=3, hidden_act="silu",
             rms_norm_eps=1e-6)
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
        teacher.lm_head.weight.mul_(12.0)

    def batch(seed):
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(MB):
            w = torch.randint(0, 999, (T + 1,), generator=g)
            with torch.no_grad():
                lp = torch.log_softmax(teacher(w[None, :-1]).float()[0], -1)
            top = lp.topk(K_, -1)
            rest = lp.exp().scatter(1, top.indices, 0.0)
            sids = torch.multinomial(rest / rest.sum(-1, keepdim=True), S_, replacement=True, generator=g)
            ids, lw = _kd_sampled(top.indices.numpy(), top.values.numpy(), sids.numpy())
            out.append({"ids": w[None, :-1], "tgt": w[1:], "kd_ids": torch.from_numpy(ids), "kd_lw": torch.from_numpy(lw)})
        return out

    student = copy.deepcopy(teacher)
    for L in student.layers:
        QT.swap_ternary(L, adapter_rank=4, quick=True)
        for n, tm in QT.tern_modules(L):
            tm.ta_packed.copy_(QT.pack3(torch.randint(-1, 2, (tm.E, tm.N, tm.ra), dtype=torch.int8)))
        for n, tm in QT.tern_modules(L):
            tm.float()
    args = types.SimpleNamespace(steps=40, theta=2.0, tau=200.0, sketch=16, rank_ratio=0.5, lr=3e-3, lr_min=3e-4,
                                 warmup=2, kd_weight=1.0, head_chunk=16, expert_chunk=4, pin=False, save_every=10 ** 9,
                                 eval_every=10 ** 9, micro=MB, seq=T, val_micro=MB, pipe_group=3, wbatch=1,
                                 ttf_store="fp32")
    return student, batch, args


def worker(rank, world, out, split):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29571")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    student, batch, args = setup()
    lo, hi = QP.pp_split(len(student.layers), world, split)[rank]
    QP.stage_model(student, lo, hi)
    tr = QP.PipeTrainer(student, args, "cpu", rank, world, lo)
    last = rank == world - 1
    recs = [tr.train_step(batch(s) if last else None) for s in range(STEPS)]
    v = tr.eval_loss(batch(999) if last else None)
    torch.save({"recs": recs, "val": v, "ts": {(lo + i, n): s for (i, n), s in tr.ts.items()},
                "w": {(lo + i, n): t.wpacked.clone() for i, L in enumerate(student.layers) for n, t in QT.tern_modules(L)}},
               os.path.join(out, f"r{rank}.pt"))
    dist.destroy_process_group()


def main():
    student, batch, args = setup()
    ref = QT.Trainer(student, args, "cpu")
    rr = [ref.train_step(batch(s)) for s in range(STEPS)]
    rv = ref.eval_loss(batch(999))
    for split in ("2,2", "1,3"):
        out = tempfile.mkdtemp()
        mp.spawn(worker, args=(2, out, split), nprocs=2)
        R = [torch.load(os.path.join(out, f"r{r}.pt"), weights_only=False) for r in range(2)]
        for s in range(STEPS):
            a, b = rr[s], R[1]["recs"][s]
            mv = {k: a["moves"][k] for k in ("exp", "dense", "adapter")}
            mb = {k: b["moves"][k] for k in ("exp", "dense", "adapter")}
            print(f"split {split} step {s + 1}: loss {a['loss']:.6f} / {b['loss']:.6f}, moves {mv} / {mb}, "
                  f"stages {b['stages']}")
            assert abs(a["loss"] - b["loss"]) <= 1e-5 * abs(a["loss"]) and mv == mb, (a, b)
        print(f"split {split} val {rv:.6f} / {R[1]['val']:.6f}")
        assert abs(rv - R[1]["val"]) <= 1e-5 * abs(rv)
        worst = 0.0
        for Rk in R:
            for (gi, n), s_ in Rk["ts"].items():
                for k, v in s_.items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        r_ = ref.ts[(gi, n)][k]
                        worst = max(worst, float((v - r_).norm() / r_.norm().clamp_min(1e-30)))
            for (gi, n), w in Rk["w"].items():
                assert torch.equal(w, QT._get(student.layers[gi], n).wpacked), (gi, n)
        print(f"split {split}: TTF state worst relative diff {worst:.2e}; trits equal")
        assert worst < 1e-5, worst
    print("q38_pipe == Trainer OK")


if __name__ == "__main__":
    main()
