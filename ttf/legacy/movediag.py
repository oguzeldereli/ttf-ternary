"""--movediag EVERY[:NB]: what each step's trit moves are worth, for TTF (kernel mode) and master alike.

On every EVERY-th step, around the real training step (nothing about training changes; the probes use their own fixed
batches, drawn once from the training data with a separate generator):
  before the step   T0 = the trits, L0 = loss on the NB loss-probe batches, G = the "true" gradient dL/dT (trit units)
                    from NB other probe batches (TTF: the kernel's captured dL/dT; master: gamma x dL/dW_q, the STE
                    gradient of the absmean-quantised weight)
  after the step    D = T1 - T0 (the moves of this step), L1 = loss on the loss probe, Lmix = the same with the old trits
                    put back (new float tail, new gamma)
Logged per diag step (and printed): the number of moves; first-order gain on the true gradient pred = -<G, D> (nats,
probe loss) and per move; precision = share of moves downhill on G (0.5 = chance); |G| of the moved entries over the
mean |G| of all entries (does it move the weights that matter?); the realized loss change of the moves, Lmix - L1, and
per move; the float part, L0 - Lmix; survive = realized / pred (curvature and interactions between the moves); the share
of moves that undo a move of the same weight within the last 1000 steps (and their share of pred); moves leaving 0 vs
moves towards 0. An "undo" is a move against the weight's net change since the diag step ~1000 steps earlier (trit
snapshots of earlier diag steps, kept on the host).
Windowed value (the one-step realized change is noisy, and a move's value can show only after the floats adapt to it):
the trits of that ~1000-step-old snapshot put back in with today's float tail, md_win_realized = L(old trits) - L1,
over md_win_net net changed entries (md_win_per_net) and md_win_age steps.
"""
from __future__ import annotations
import collections
import torch

from .flip import KernelTernaryLinear
from .kernel import pack_rows, unpack_rows
from .master import MasterTernaryLinear
from .bitlinear import STATE


def _get_batch(data, bs, seq_len, device, gen):
    ix = torch.randint(len(data) - seq_len - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype("int64")) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype("int64")) for i in ix.tolist()])
    return x.to(device), y.to(device)


class MoveDiag:
    def __init__(self, model, data, tc, device, mode, every, nb=8, seed=424242, undo_window=1000):
        self.model, self.mode, self.every, self.device = model, mode, every, device
        self.dt = device.split(":")[0]
        cls = KernelTernaryLinear if mode != "master" else MasterTernaryLinear
        self.layers = [m for m in model.modules() if isinstance(m, cls)]
        gen = torch.Generator().manual_seed(seed)
        self.gb = [_get_batch(data, tc.batch_size, tc.seq_len, device, gen) for _ in range(nb)]
        self.lb = [_get_batch(data, tc.batch_size, tc.seq_len, device, gen) for _ in range(nb)]
        self.undo_window = undo_window
        self.hist = collections.deque()      # (step, trits on the host) of earlier diag steps, for the undo share
        self.active = False

    def on(self, step):
        return self.every > 0 and step > 0 and step % self.every == 0

    @torch.no_grad()
    def _trits(self):
        if self.mode == "master":
            return [m.ternary_weight()[0].to(torch.int8) for m in self.layers]
        return [unpack_rows(m.wpacked, m.K)[:, :m.K].to(torch.int8) for m in self.layers]

    @torch.no_grad()
    def _loss(self, batches):
        tot = 0.0
        for x, y in batches:
            with torch.autocast(device_type=self.dt, dtype=torch.bfloat16):
                _, loss = self.model(x, y)
            tot += loss.item()
        return tot / len(batches)

    def _true_grad(self):
        nb = len(self.gb)
        if self.mode == "master":
            for m in self.layers:
                m.weight.grad = None
            for x, y in self.gb:
                with torch.autocast(device_type=self.dt, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                (loss / nb).backward()
            G = []
            for m in self.layers:
                _, gam = m.ternary_weight()
                G.append(m.weight.grad.float() * gam.float())
                m.weight.grad = None
        else:
            saved = [(m.capture, getattr(m, "on_grad", None), getattr(m, "cap_accum", False)) for m in self.layers]
            for m in self.layers:
                m.capture = True; m.on_grad = None; m.cap_accum = False; m.gw = None
            G = [None] * len(self.layers)
            for x, y in self.gb:
                with torch.autocast(device_type=self.dt, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                (loss / nb).backward()
                for i, m in enumerate(self.layers):
                    g = m.gw[:, :m.K].float(); m.gw = None
                    G[i] = g if G[i] is None else G[i].add_(g)
            for m, (c, og, ca) in zip(self.layers, saved):
                m.capture, m.on_grad, m.cap_accum = c, og, ca
        for p in self.model.parameters():        # the probes' float gradients must not reach the optimizer
            p.grad = None
        return G

    def pre(self, step):
        """call before the training step's zero_grad / forward"""
        was = STATE.updates_enabled
        STATE.updates_enabled = False
        self.T0 = self._trits()
        self.P0 = [m.wpacked.clone() for m in self.layers] if self.mode != "master" else None
        self.L0 = self._loss(self.lb)
        self.G = self._true_grad()
        STATE.updates_enabled = was
        self.active = True

    @torch.no_grad()
    def post(self, step):
        """call after the step's updates (trits and float tail); returns the record to log"""
        if not self.active:
            return None
        self.active = False
        while len(self.hist) > 1 and step - self.hist[1][0] >= self.undo_window:
            self.hist.popleft()
        was = STATE.updates_enabled
        STATE.updates_enabled = False
        T1 = self._trits()
        L1 = self._loss(self.lb)
        # the old trits back in, with the new float tail (and master's new gamma)
        if self.mode == "master":
            for m, t in zip(self.layers, self.T0):
                m.T = t
            Lmix = self._loss(self.lb)
            for m in self.layers:
                m.T = None
        else:
            P1 = [m.wpacked.clone() for m in self.layers]
            for m, p in zip(self.layers, self.P0):
                m.wpacked.copy_(p)
            Lmix = self._loss(self.lb)
            for m, p in zip(self.layers, P1):
                m.wpacked.copy_(p)
            del P1
        old = self.hist[0] if self.hist and step - self.hist[0][0] >= self.undo_window // 2 else None
        Lwin = None; win_net = 0
        if old is not None:                       # the trits of ~undo_window steps ago, today's float tail
            if self.mode == "master":
                for m, t in zip(self.layers, old[1]):
                    m.T = t.to(self.device)
                Lwin = self._loss(self.lb)
                for m in self.layers:
                    m.T = None
            else:
                P1 = [m.wpacked.clone() for m in self.layers]
                for m, t in zip(self.layers, old[1]):
                    m.wpacked.copy_(pack_rows(t.to(self.device)))
                Lwin = self._loss(self.lb)
                for m, p in zip(self.layers, P1):
                    m.wpacked.copy_(p)
                del P1
            win_net = sum(int((t1 != t.to(self.device)).sum().item()) for t1, t in zip(T1, old[1]))
        STATE.updates_enabled = was
        n = 0; pred = 0.0; down = 0; gm_moved = 0.0; gm_all = 0.0; nall = 0
        undo_n = 0; undo_pred = 0.0; leave0 = 0; to0 = 0
        for i, (t0, t1, g) in enumerate(zip(self.T0, T1, self.G)):
            d = (t1.to(torch.int16) - t0.to(torch.int16)).to(torch.int8)
            mv = d != 0
            k = int(mv.sum().item())
            gd = (g * d.float())
            gm_all += g.abs().sum().item(); nall += g.numel()
            if k:
                n += k
                pred -= gd.sum().item()
                down += int((gd[mv] < 0).sum().item())
                gm_moved += g.abs()[mv].sum().item()
                if old is not None:
                    net = t0.to(torch.int16) - old[1][i].to(self.device, non_blocking=True).to(torch.int16)
                    und = mv & (net * d.to(torch.int16) < 0)
                    undo_n += int(und.sum().item()); undo_pred -= gd[und].sum().item()
                leave0 += int((mv & (t0 == 0)).sum().item()); to0 += int((mv & (t1 == 0)).sum().item())
        self.hist.append((step, [t.cpu() for t in T1]))
        nw = sum(m.N * m.K for m in self.layers)
        rec = dict(movediag=True, md_moves=n, md_move_frac=n / nw, md_pred=pred, md_pred_per_move=pred / max(n, 1),
                   md_precision=down / max(n, 1), md_g_ratio=(gm_moved / max(n, 1)) / (gm_all / max(nall, 1)),
                   md_realized=Lmix - L1, md_realized_per_move=(Lmix - L1) / max(n, 1), md_float=self.L0 - Lmix,
                   md_total=self.L0 - L1, md_survive=(Lmix - L1) / pred if pred else float("nan"),
                   md_undo_frac=undo_n / max(n, 1) if old is not None else float("nan"),
                   md_undo_pred_share=undo_pred / pred if (pred and old is not None) else float("nan"),
                   md_leave0_frac=leave0 / max(n, 1), md_to0_frac=to0 / max(n, 1), md_L0=self.L0,
                   md_win_realized=(Lwin - L1) if Lwin is not None else float("nan"),
                   md_win_net=win_net, md_win_per_net=(Lwin - L1) / max(win_net, 1) if Lwin is not None else float("nan"),
                   md_win_age=step - old[0] if old is not None else 0)
        self.T0 = self.G = self.P0 = None
        return rec
