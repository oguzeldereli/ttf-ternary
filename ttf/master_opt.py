"""Master weights with one ingredient taken away at a time (which part of master's update is the part we lack?).

AdamX: AdamW (decoupled weight decay, bias correction, eps 1e-8, the same math as torch.optim.AdamW) with two optional
restrictions on the groups that set them (the latent master weights):
  factv   the second moment kept factored, Adafactor-style: row and column EMAs of g^2, v_ij ~ R_i C_j / mean(R)
          (our rule's normalization) instead of one v per weight
  gate    the update applied only where this batch's gradient agrees in sign with m (our sign gate)
  rank r  the first moment kept at rank r, updated like our momentum (one subspace-iteration step per update:
          V' = qr(b1 V (U^T U) + (1 - b1) g^T U), U' = b1 U (V^T V') + (1 - b1) g V')

latent_ops: what happens to the latent between updates (applied after every optimizer step):
  clamp c   |W| <= c * gamma: no depth past the trit's centre (c = 1) or none past the edge of the +-1 bin (c ~ 0.5)
  leak tau  W -= (W - t gamma) / tau: the offset from the trit's centre (master's sub-threshold memory) decays with a
            time constant of tau steps (our momentum's memory is ~220-390 steps)
  snap      when a trit changes, W is set to the new trit's centre t * gamma: no leftover offset after a crossing
            (a weight that just crossed no longer sits at the boundary, so reversing it costs a full half-bin)
gamma = mean|W| of the matrix (or the frozen value, --m_gfix).
"""
import math
import torch


class AdamX(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, factv=False, rank=0, gate=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, factv=factv, rank=rank,
                                      gate=gate))

    def load_state_dict(self, sd):
        """also loads a torch AdamW state (branching master's checkpoint): its moments become m (or its best rank-r
        part) and v (or v's row / column means); this optimizer's own restrictions and betas are kept"""
        keep = [{k: g[k] for k in ("factv", "rank", "betas", "gate")} for g in self.param_groups]
        super().load_state_dict(sd)
        for g, k in zip(self.param_groups, keep):
            g.update(k)
            for p in g["params"]:
                st = self.state.get(p)
                if not st or "exp_avg" not in st: continue
                m, v = st.pop("exp_avg").float().clone(), st.pop("exp_avg_sq").float().clone(); t = int(st.pop("step"))
                st.clear(); st["t"] = t
                if g["rank"] and p.dim() == 2:
                    r = min(g["rank"], *p.shape)
                    Uf, Sf, Vh = torch.linalg.svd(m, full_matrices=False)
                    st["U"] = Uf[:, :r] * Sf[:r]; st["V"] = Vh[:r].T.contiguous()
                else:
                    st["m"] = m
                if g["factv"] and p.dim() == 2:
                    st["R"] = v.mean(1); st["C"] = v.mean(0)
                else:
                    st["v"] = v

    @torch.no_grad()
    def step(self, closure=None):
        for gr in self.param_groups:
            b1, b2 = gr["betas"]; lr, eps, wd = gr["lr"], gr["eps"], gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None: continue
                g = p.grad.float(); st = self.state[p]
                if not st:
                    st["t"] = 0
                    if gr["rank"] and p.dim() == 2:
                        r = min(gr["rank"], *p.shape)
                        st["U"] = torch.zeros(p.shape[0], r, device=p.device)
                        st["V"] = torch.linalg.qr(torch.randn(p.shape[1], r, device=p.device))[0]
                    else:
                        st["m"] = torch.zeros_like(g)
                    if gr["factv"] and p.dim() == 2:
                        st["R"] = torch.zeros(p.shape[0], device=p.device); st["C"] = torch.zeros(p.shape[1], device=p.device)
                    else:
                        st["v"] = torch.zeros_like(g)
                st["t"] += 1; t = st["t"]
                if wd: p.mul_(1 - lr * wd)
                if "U" in st:
                    U, V = st["U"], st["V"]
                    Vn = torch.linalg.qr(b1 * V @ (U.T @ U) + (1 - b1) * g.T @ U)[0] if t > 1 else \
                        torch.linalg.qr(g.T @ torch.randn(g.shape[0], V.shape[1], device=g.device))[0]
                    U = b1 * U @ (V.T @ Vn) + (1 - b1) * g @ Vn
                    st["U"], st["V"] = U, Vn
                    m = U @ Vn.T
                else:
                    st["m"].mul_(b1).add_(g, alpha=1 - b1); m = st["m"]
                if "R" in st:
                    g2 = g * g
                    st["R"].mul_(b2).add_(g2.mean(1), alpha=1 - b2); st["C"].mul_(b2).add_(g2.mean(0), alpha=1 - b2)
                    v = st["R"][:, None] * st["C"][None, :] / st["R"].mean().clamp_min(1e-30)
                else:
                    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2); v = st["v"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                u = (m / bc1) / ((v / bc2).sqrt() + eps)
                if gr["gate"]: u = u * (m.sign() == g.sign())
                p.add_(u, alpha=-lr)
        return None


@torch.no_grad()
def latent_ops(layers, state, clamp=0.0, leak=0.0, snap=False, lag=0.0, step=0, opt=None):
    """after an optimizer step: clamp / leak / snap the latent master weights (see the module docstring); with opt
    (--m_snap_mreset) a snapped weight's first moment is zeroed too (AdamW exp_avg / AdamX m), so no carried momentum
    pushes it on from the new centre"""
    for i, l in enumerate(layers):
        w = l.weight
        gam = l.gamma_fixed if getattr(l, "gamma_fixed", None) is not None else w.abs().mean().clamp_min(1e-5)
        t = (w / gam).round().clamp_(-1, 1)
        if clamp:
            w.clamp_(-clamp * gam, clamp * gam)
        if leak:
            w.sub_((w - t * gam) / leak)
        if snap:
            prev = state.get(i)
            if prev is not None:
                ch = t != prev
                w.copy_(torch.where(ch, t * gam, w))
                if opt is not None:
                    st_ = opt.state.get(w, {})
                    for k_ in ("exp_avg", "m"):
                        if k_ in st_:
                            st_[k_].masked_fill_(ch, 0.0)
            state[i] = t.to(torch.int8)
        if lag:
            # --m_lag q: the forward's trits T follow round(W / gamma) only stochastically: each weight whose latent says
            # another trit changes with probability q per step (our rule fires at a rate, not at a threshold)
            if l.T is None:
                l.T = t.to(torch.int8)
            else:
                g_ = torch.Generator(device=w.device).manual_seed(5_000_000 + 131 * step + i)
                fire = (t.to(torch.int8) != l.T) & (torch.rand(t.shape, generator=g_, device=w.device) < lag)
                l.T = torch.where(fire, t.to(torch.int8), l.T)
