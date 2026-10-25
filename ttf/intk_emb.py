"""--intk-emb E:H[:G]: the token embedding and output head as low-bit integer codes trained by TTF (no float master,
no Adam state for them): W = lo + s * q per group of G entries along each row, q in {0 .. 2^bits - 1} (E bits for the
embedding, H for the head; e.g. 2:3; tied: one matrix, E bits), lo and s per group (fp32, trained in the float tail with
the norm gains). Started from the round-to-nearest codes of the (converted, folded) float matrices, or of the random
init (training from scratch: every weight matrix of the model is then TTF-trained).

TTF moves a code by one level when its accumulator reaches theta, as it moves a trit; the gradient it sees is dL/dq =
dL/dW * s (one level is one step of s).

--intk-update factored (the default): as the ternary layers' --dw factored update, the weight gradient is never formed.
  - head: the loss runs in token chunks, each chunk's logits from the codes dequantized in vocabulary chunks (no [V, d]
    matrix); dL/dlogits of the micro-batch (bf16 [T, V]) and its input x are the gradient's factors, g = gy^T x.
  - embedding: the forward gathers the dequantized rows of the batch's tokens; the gradient is the scatter of dL/dh into
    those rows, g = P^T G (P the one-hot rows of the distinct tokens, G their summed dL/dh): only those rows are touched.
  - a tied matrix sums both (the update runs once both factors of the micro-batch are in).
  The update (_intk_update) is _ts_layer_factored's (micro-batch u into U on the step's fixed basis, the factored v from
  a k-column sketch, the basis turn, firing and spend on the last micro-batch) with every [N, .] product taken in row
  chunks of ROWS: U is stored per row chunk (each its own codec scale), so the largest temporaries are [ROWS, d]. The
  group parameters' gradients come from the same factors (per vocabulary chunk, gy_c^T x: one more head-sized GEMM).
--intk-update dense (the 25 Oct path): the dequantized matrix formed once per optimizer step as a bf16 leaf, its
  gradient summed in fp32 over the micro-batches, then one dense update per step (ts_layer / ttf_rows).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntKMatrix(nn.Module):
    def __init__(self, W, bits, g=64):
        super().__init__()
        N, K = W.shape
        assert K % g == 0, (K, g)
        self.N, self.K, self.g, self.bits, self.levels = N, K, g, bits, 2 ** bits - 1
        with torch.no_grad():
            Wf = W.float().reshape(N, K // g, g)
            lo = Wf.amin(-1); s = ((Wf.amax(-1) - lo) / self.levels).clamp_min(1e-12)
            q = ((Wf - lo[..., None]) / s[..., None]).round().clamp(0, self.levels)
        self.register_buffer("codes", q.reshape(N, K).to(torch.int8))
        self.lo, self.s = nn.Parameter(lo.contiguous()), nn.Parameter(s.contiguous())
        self.track, self._ts_host, self._W = False, False, None
        self._parts, self.n_parts = [], 1            # --intk-update factored: the micro-batch's gradient factors (tied: 2)
        self._ts_state = self._ts_cfg = None         # (set by the trainer: lr_state, ts_cfg)
        self.theta_mult = 1.0                        # --intk-theta-mult: the codes' theta x this (inf: codes frozen)

    def deq_rows(self, i, j, dtype=torch.bfloat16):
        """rows i:j of W = lo + s q (no autograd)"""
        q = self.codes[i:j].reshape(j - i, self.K // self.g, self.g).float()
        lo, s = self.lo.detach()[i:j], self.s.detach()[i:j]
        return (lo[..., None] + s[..., None] * q).reshape(j - i, self.K).to(dtype)

    def deq_idx(self, idx, dtype=torch.bfloat16):
        """the rows idx of W (the embedding's lookup)"""
        q = self.codes[idx].reshape(-1, self.K // self.g, self.g).float()
        return (self.lo.detach()[idx][..., None] + self.s.detach()[idx][..., None] * q).reshape(-1, self.K).to(dtype)

    def add_part(self, part):
        """--intk-update factored: one use's gradient factors of this micro-batch; the update runs once all n_parts are in"""
        self._parts.append(part)
        if len(self._parts) >= self.n_parts:
            parts, self._parts = self._parts, []
            if self._ts_state is not None and self._ts_cfg is not None and math.isfinite(self.theta_mult):
                _intk_update(self, parts, self._ts_state, self._ts_cfg)

    def dequant(self):
        q = self.codes.reshape(self.N, self.K // self.g, self.g).float()
        return (self.lo[..., None] + self.s[..., None] * q).reshape(self.N, self.K)

    DEQ_ROWS = 8192

    def dequant_bf16(self):
        """dequant() rounded to bf16, formed in row chunks (no autograd): the same values without the whole fp32 matrix
        (27B embedding: a 5 GB fp32 transient before its 2.5 GB bf16 copy, on every eval batch through the head)"""
        out = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.codes.device)
        for i in range(0, self.N, self.DEQ_ROWS):
            j = min(self.N, i + self.DEQ_ROWS)
            out[i:j] = self.deq_rows(i, j)
        return out

    def leaf(self):
        """the step's dequantized matrix (bf16): a leaf while training, else a plain tensor. Each micro-batch's gradient
        (bf16, rounded once) is moved by a post-accumulate hook into an fp32 sum (self._G), so the step's sum is fp32 and
        no bf16 .grad accumulates (at 27B: 2.5 GB per matrix instead of an fp32 leaf's 5 GB plus a bf16 copy)"""
        if not torch.is_grad_enabled():
            return self.dequant_bf16()
        if self._W is None:
            self._W = self.dequant_bf16().requires_grad_(True)
            self._G = None

            def _acc(p):
                g = p.grad.float()
                self._G = g if self._G is None else self._G.add_(g)
                p.grad = None
            self._W.register_post_accumulate_grad_hook(_acc)
        return self._W

    @torch.no_grad()
    def step(self, ts_layer, state, c):
        """after the step's last backward: the group parameters' gradients and the TTF update of the codes"""
        G_ = getattr(self, "_G", None)
        self._W = self._G = None
        if G_ is None:
            return
        import torch.distributed as dist
        if getattr(self, "dp_reduce", False) and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:   # data parallel: the codes
            dist.all_reduce(G_); G_.div_(dist.get_world_size())   # move on the mean gradient (the same on every rank)
        gW = G_.reshape(self.N, self.K // self.g, self.g)
        q = self.codes.reshape(self.N, self.K // self.g, self.g).float()
        for p, gp in ((self.lo, gW.sum(-1)), (self.s, (gW * q).sum(-1))):
            gp = gp.to(p.dtype)                          # (bf16 lo / s on a bf16 / lean tail)
            p.grad = gp if p.grad is None else p.grad + gp
        if not math.isfinite(self.theta_mult):        # --intk-theta-mult inf: only the group parameters train
            return
        c = dict(c, theta=c["theta"] * self.theta_mult)
        gq = (gW * self.s.detach()[..., None]).reshape(self.N, self.K)
        del gW, q, G_
        if self.N * self.K > CHUNKED_ABOVE[0]:
            ttf_rows(self, gq, state, c)
        else:
            ts_layer(self, gq, state, dict(c, micro=1))


CHUNKED_ABOVE = [400_000_000]   # entries: larger matrices (27B: 248320 x 5120) take ttf_rows (bounded temporaries)


@torch.no_grad()
def ttf_rows(l, g, state, c, rows=16384):
    """update.ts_layer's dense update (one call per step) for int-k codes in row chunks: the temporaries are
    [rows, K] instead of [N, K] (at 27B's 248320 x 5120 the full ones were ~30 GB). The same steps: v = R C / mean R
    (row / column EMAs of g^2, decay b2, bias-corrected), u = -lr g / (sqrt v + 1e-8), the rank-r accumulator A = U V^T
    (leak, u on the basis, then the basis turned: update._acc_turn), a code moves one level where |A| >= theta (within 0 .. levels),
    each move spends theta. V's first basis comes from a chunk-wise random projection (not ts_layer's single draw)."""
    from .update import _orth
    N, K = g.shape
    lr, theta, tau, b2 = c["lr_ratio"], c["theta"], c["tau"], c["b2"]
    st = state.setdefault("_ts", {}); key = id(l)
    t = st.get(("t", key), 0) + 1; st[("t", key)] = t
    R = torch.empty(N, device=g.device); Cs = torch.zeros(K, device=g.device)
    for i in range(0, N, rows):
        g2 = g[i:i + rows].square(); R[i:i + rows] = g2.mean(1); Cs += g2.sum(0); del g2
    C = Cs / N
    if ("v", key) in st:
        R0, C0 = st[("v", key)]; R = b2 * R0 + (1 - b2) * R; C = b2 * C0 + (1 - b2) * C
    else:
        R, C = (1 - b2) * R, (1 - b2) * C
    st[("v", key)] = (R, C)
    rs = (R / R.mean().clamp_min(1e-30) / (1 - b2 ** t)).sqrt(); cs = C.sqrt()   # sqrt(v_ij) = rs_i cs_j

    def u_of(i):
        return -g[i:i + rows] / (rs[i:i + rows, None] * cs[None, :] + 1e-8) * lr

    ra = min(int(getattr(l, "ts_rank", None) or c["r"]), N, K)
    if key not in state:
        gen = torch.Generator(device=g.device).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        X = torch.zeros(K, ra, device=g.device)
        for i in range(0, N, rows):
            u = u_of(i); X += u.T @ torch.randn(u.shape[0], ra, device=g.device, generator=gen); del u
        V = _orth(X); U = torch.empty(N, ra, device=g.device)
        for i in range(0, N, rows):
            U[i:i + rows] = u_of(i) @ V
    else:
        U, V = state[key]
        U = U * (1 - lr / tau)
        X = V @ (U.T @ U)                              # update._acc_turn, a one-batch step: the turn from U_0
        for i in range(0, N, rows):
            X += u_of(i).T @ U[i:i + rows]
        Vn = _orth(X); del X
        P = V.T @ Vn
        for i in range(0, N, rows):
            U[i:i + rows] = U[i:i + rows] @ P + u_of(i) @ Vn
        V = Vn
    nf = torch.zeros((), device=g.device, dtype=torch.long)
    for i in range(0, N, rows):
        A = U[i:i + rows] @ V.T; w0 = l.codes[i:i + rows]
        d = A.sign(); nw = w0.float() + d
        fire = (A.abs() >= theta) & (nw >= 0) & (nw <= l.levels); del A, nw
        D = torch.where(fire, d, torch.zeros_like(d)); del d
        w0.add_(D.to(torch.int8)); nf += fire.sum(); del fire
        U[i:i + rows] -= c["spend"] * (theta * (D @ V)); del D
    state[key] = (U, V)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + N * K


INTK_FACTORED = [True]   # --intk-update factored (default) | dense
ROWS = [8192]            # rows per chunk of the factored update and of the head's vocabulary loops
TOK_BYTES = [1 << 29]    # the int-k head loss: tokens per chunk such that the chunk's fp32 logits stay below this
DENSE_G = [None]         # the int-k head's gradient to the update: None (by size), True (dense [V, d]), False (factors)


class _Parts:
    """the factors of one micro-batch's gradient g = sum over parts of gy_p^T x_p: dense parts ("d", gy [T, N] bf16,
    x [T, K] bf16) (the head) and sparse parts ("s", u [n] distinct rows, G [n, K] fp32) (the embedding: gy = the
    one-hot rows of u). The update reads g only through R(M) = g M ([N, c], taken in row chunks) and L(W) = g^T W
    ([K, c], W given in row chunks)."""

    def __init__(self, parts, rows=None):
        self.p, self.rows = parts, rows or ROWS[0]       # (rows: the store's chunking, which a checkpoint fixes)

    def prep(self, M):
        """x_p M for every part (M [K, c]); for a "g" part (the gradient itself, ("g", G [N, K] fp32, None): the head when
        its dense gradient is smaller than its factors) M itself"""
        from .update import _mmb
        return [M if k == "g" else _mmb(x, M) if k == "d" else x @ M for k, _, x in self.p]

    def R_rows(self, xm, i, j):
        """rows i:j of g M from prep(M)"""
        from .update import _mmb
        out = None
        for (k, a, _), y in zip(self.p, xm):
            if k == "g":
                o = a[i:j] @ y
            elif k == "d":
                o = _mmb(a[:, i:j].T, y)
            else:
                sel = (a >= i) & (a < j)
                o = torch.zeros(j - i, y.shape[1], device=y.device, dtype=torch.float32)
                o.index_add_(0, a[sel] - i, y[sel].float())
            out = o if out is None else out.add_(o)
        return out

    def R(self, M, N):
        xm = self.prep(M)
        return torch.cat([self.R_rows(xm, i, min(N, i + self.rows)) for i in range(0, N, self.rows)])

    def L(self, rows_of, N, c, dev):
        """g^T W = sum_p x_p^T (gy_p W), W's rows i:j given by rows_of(i, j) ([j - i, c])"""
        from .update import _mmb
        acc = [torch.zeros(a.shape[1] if k == "g" else a.shape[0], c, device=dev, dtype=torch.float32)
               for k, a, _ in self.p]
        for i in range(0, N, self.rows):
            j = min(N, i + self.rows); W = rows_of(i, j)
            for (k, a, _), A in zip(self.p, acc):
                if k == "g":
                    A += a[i:j].T @ W.float()
                elif k == "d":
                    A += _mmb(a[:, i:j], W)
                else:
                    sel = (a >= i) & (a < j)
                    A[sel] = W[a[sel] - i].float()
        out = None
        for (k, _, x), A in zip(self.p, acc):
            o = A if k == "g" else _mmb(x.T, A) if k == "d" else x.T @ A
            out = o if out is None else out.add_(o)
        return out


def _st_get(t, dev, off):
    return t.to(dev, non_blocking=True) if off else t


def _st_put(old, x, off):
    """a stored tensor written back (--ttf-offload: into its pinned host buffer, reused across steps)"""
    if not off:
        return x
    from .update import _PINNED
    if old is not None and old.device.type == "cpu" and old.shape == x.shape and old.dtype == x.dtype and _PINNED.owns(old):
        return old.copy_(x, non_blocking=True)
    return _PINNED.empty(x.shape, x.dtype).copy_(x, non_blocking=True)


class _Store:
    """an int-k matrix's accumulator: V and U in row chunks of ROWS, each encoded by the --ttf-store codec on its own (its
    own per-column scale), as one flat tuple state[key] = (V codes.., U chunk 0 codes.., U chunk 1 codes.., ...) (so the
    checkpoint code saves it as any other state tuple); state["_ts"][("f", key)] = (ROWS, codes per tensor)"""

    def __init__(self, l, state, key, sa, off, dev):
        self.l, self.state, self.key, self.sa, self.off, self.dev = l, state, key, sa, off, dev
        st = state["_ts"]
        if key in state and ("f", key) not in st:          # a dense-path (--intk-update dense) checkpoint: U whole
            from .update import _load_pair
            U, V = _load_pair(tuple(x.to(dev) for x in state[key]))
            del state[key]
            self.rows, self.ne = ROWS[0], self._ne()
            self.t = [None] * (self.ne * (1 + self.nch))
            self.put_v(V)
            for c in range(self.nch):
                self.put_u(c, U[c * self.rows:(c + 1) * self.rows])
            self.flush(); del U
        if key in state:
            self.rows, self.ne = st[("f", key)]
            self.t = list(state[key])
        else:
            self.rows, self.ne, self.t = ROWS[0], self._ne(), None

    def _ne(self):
        return len(self.sa(torch.zeros(2, 2))) if self.sa else 1

    @property
    def nch(self):
        return (self.l.N + self.rows - 1) // self.rows

    def rng(self, c):
        return c * self.rows, min(self.l.N, (c + 1) * self.rows)

    def _dec(self, ts):
        from .update import _decode
        ts = [_st_get(x, self.dev, self.off) for x in ts]
        return ts[0].float() if len(ts) == 1 else _decode(ts[0], ts[1])

    def _enc(self, X):
        return self.sa(X) if self.sa else (X,)

    def v(self):
        return self._dec(self.t[:self.ne])

    def u(self, c):
        a = self.ne * (1 + c)
        return self._dec(self.t[a:a + self.ne])

    def put_v(self, V):
        self.t[:self.ne] = [_st_put(o, x, self.off) for o, x in zip(self.t[:self.ne], self._enc(V))]

    def put_u(self, c, U):
        a = self.ne * (1 + c)
        self.t[a:a + self.ne] = [_st_put(o, x, self.off) for o, x in zip(self.t[a:a + self.ne], self._enc(U))]

    def flush(self):
        self.state[self.key] = tuple(self.t)
        self.state["_ts"][("f", self.key)] = (self.rows, self.ne)


@torch.no_grad()
def _intk_update(l, parts, state, c):
    """--intk-update factored: one micro-batch's TTF update of an int-k matrix from its gradient factors (see _Parts):
    _ts_layer_factored's maths (its torch branch: the factored v from a k-column sketch of g, u = -lr diag(1/a) g
    diag(1/b), with --accum M > 1 each micro-batch's u V into U on the step's fixed basis, the turn, firing and spend on
    the last), with every [N, .] product in row chunks; a code moves one level within 0 .. levels"""
    from .update import _orth, _dpm, _store_codec, _host_get, _host_put, _layer_gen, _step_scale
    from .schedule import _theta_at
    P = _Parts(parts)
    N, K = l.N, l.K
    r, lr_ratio, theta, tau, b2 = c["r"], c["lr_ratio"], c["theta"], c["tau"], c["b2"]
    r = getattr(l, "ts_rank", None) or r
    if c.get("theta_sched"):
        theta = _theta_at(c["theta_sched"], c.get("progress", 0.0))
    theta = theta * l.theta_mult
    M_ = int(c.get("micro") or 1)
    final_ = getattr(l, "_ts_final", True) if M_ > 1 else True
    st = state.setdefault("_ts", {})
    dev = parts[0][1].device
    off = bool(c.get("offload", False)) if getattr(l, "_ts_host", True) else False
    sa = _store_codec(c.get("store_a", ""))
    key = id(l)
    old_ = key in state and ("f", key) not in st          # a dense-path state: its count was in steps, now micro-batches
    S = _Store(l, state, key, sa, off, dev)
    P.rows = S.rows
    if old_:
        st[("t", key)] = st.get(("t", key), 0) * M_
    t = st.get(("t", key), 0) + 1; st[("t", key)] = t
    lk = 1 - lr_ratio / tau
    k_ = int(c.get("factored") or 0) or 64
    ts_ = (t - 1) // M_
    gen = _layer_gen(l, dev).manual_seed(11_000_000 + 2 * getattr(l, "_ts_idx", 0) + 1_000_003 * ts_)
    Sr = torch.randn(K, k_, device=dev, generator=gen) / k_ ** 0.5
    Sc = torch.randn(k_, N, device=dev, generator=gen) / k_ ** 0.5
    gS = _dpm(P.R(Sr, N))                                                     # g Sr  [N, k]
    Sg = _dpm(P.L(lambda i, j: Sc.T[i:j], N, k_, dev).T.contiguous())         # Sc g  [k, K]
    rsq, csq = gS.square().sum(1), Sg.square().sum(0)                         # ~ row / column sums of g^2
    if M_ > 1:
        r0, c0 = rsq * (M_ * M_ / K), csq * (M_ * M_ / N)
        if ("v", key) in st and ts_ > 0:
            R0, C0 = _host_get(st, ("v", key), dev, off)
            bc_ = 1 - b2 ** ts_
            r0 = b2 * R0 / bc_ + (1 - b2) * r0; c0 = b2 * C0 / bc_ + (1 - b2) * c0; del R0, C0
        m_ = r0.mean().clamp_min(1e-30)
        PS, SP, rs, cs, ds, es = st.pop(("mx", key), None) or (0, 0, 0, 0, 0, 0)
        if torch.is_tensor(PS):
            PS, SP = PS.float(), SP.float()
        d1, d0 = rsq / K, csq / N
        rs = rs + d1 + (2.0 / K) * (gS * PS).sum(1) if torch.is_tensor(PS) else d1
        cs = cs + d0 + (2.0 / N) * (Sg * SP).sum(0) if torch.is_tensor(SP) else d0
        ds, es = ds + d1, es + d0
        PS, SP = PS + gS, SP + Sg
        if final_:
            rs, cs = torch.maximum(rs, ds / M_), torch.maximum(cs, es / M_)
            if ("v", key) in st:
                R, C = _host_get(st, ("v", key), dev, off); R = b2 * R + (1 - b2) * rs; C = b2 * C + (1 - b2) * cs
            else:
                R, C = (1 - b2) * rs, (1 - b2) * cs
            _host_put(st, ("v", key), (R, C), off)
            turn_ = (SP, PS) + _step_scale(R, C, b2, ts_ + 1)   # Sc sum g [k, K], (sum g) Sr [N, k]: the turn's sketches
        else:
            st[("mx", key)] = (PS.to(torch.bfloat16), SP.to(torch.bfloat16), rs, cs, ds, es)
        del PS, SP
        a = (r0 / m_).sqrt_(); b = c0.sqrt()
    else:
        if ("v", key) in st:
            R, C = _host_get(st, ("v", key), dev, off); R = b2 * R + (1 - b2) * rsq / K; C = b2 * C + (1 - b2) * csq / N
        else:
            R, C = (1 - b2) * rsq / K, (1 - b2) * csq / N
        _host_put(st, ("v", key), (R, C), off)
        a = (R / R.mean().clamp_min(1e-30) / (1 - b2 ** t)).sqrt(); b = C.sqrt()
    del gS, Sg
    ia = 1.0 / torch.maximum(a, a.max() * 1e-12 + 1e-30)
    ib = 1.0 / torch.maximum(b, b.max() * 1e-12 + 1e-30)
    if c.get("start") and c.get("step", 0) < c["start"]:
        return
    ra = min(r, N, K)

    def r_term(W):          # rows of ia * (g (ib W)) = _fg_mm_r(fgb, ia, W), by chunk
        xm = P.prep(ib[:, None] * W)
        return lambda i, j: ia[i:j, None] * P.R_rows(xm, i, j)

    def l_term(rows_of):    # ib * (g^T (ia W)) = _fg_mm_l(fgb, ia, W)
        return ib[:, None] * P.L(lambda i, j: ia[i:j, None] * rows_of(i, j), N, ra, dev)

    if S.t is None:         # the first call: V from a random projection, U = u V
        g_ = torch.Generator(device=dev).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        Rn = [torch.randn(j - i, ra, device=dev, generator=g_) for i, j in map(S.rng, range(S.nch))]
        V = _orth(-lr_ratio * _dpm(l_term(lambda i, j: Rn[i // S.rows])))
        del Rn
        S.t = [None] * (S.ne * (1 + S.nch))
        rt = r_term(V)
        Us = [(-lr_ratio) * _dpm(rt(*S.rng(cc))) for cc in range(S.nch)]
        if not final_:
            S.put_v(V)
            for cc, U in enumerate(Us):
                S.put_u(cc, U)
            S.flush(); return
        _fire(l, S, Us, V, theta, c, st)
        return
    # update._acc_turn by row chunk: a one-batch step turns from U_0 (exact); with micro-batches every micro-batch's u
    # enters on the step's basis and the basis turns on the last, from U and the step's sketches
    V = S.v()
    if M_ == 1:
        Us = [S.u(cc) * lk for cc in range(S.nch)]
        UtU = sum(U.T @ U for U in Us)
        Vn = _orth(V @ UtU - lr_ratio * _dpm(l_term(lambda i, j: Us[i // S.rows]))); del UtU
        rt = r_term(Vn); P_ = V.T @ Vn
        Us = [U @ P_ - lr_ratio * _dpm(rt(*S.rng(cc))) for cc, U in enumerate(Us)]
        _fire(l, S, Us, Vn, theta, c, st)
        return
    rt = r_term(V)
    Us = []
    for cc in range(S.nch):
        U = S.u(cc)
        if (t - 1) % M_ == 0:
            U = U * lk
        U.add_(rt(*S.rng(cc)), alpha=-lr_ratio)     # (data parallel: this rank's term, averaged on the last micro-batch)
        if final_:
            Us.append(_dpm(U))
        else:
            S.put_u(cc, U)
    if not final_:
        S.flush(); return
    SP_, PS_, ia_, ib_ = turn_
    UtU = sum(U.T @ U for U in Us)
    Y = sum(Sc[:, i:j] @ (ia_[i:j, None] * U) for U, (i, j) in zip(Us, map(S.rng, range(S.nch))))
    S_ = (-lr_ratio) * ib_[:, None] * (SP_.float().T @ Y); del Y          # update._sketch_cross
    S_ = S_ - V @ (V.T @ S_)
    Vn = _orth(V @ UtU + S_); del S_, UtU
    P_ = V.T @ Vn
    Z = Sr.T @ (ib_[:, None] * (Vn - V @ P_))                           # update._sketch_mass
    Us = [U @ P_ - lr_ratio * ia_[i:j, None] * (PS_[i:j].float() @ Z) for U, (i, j) in zip(Us, map(S.rng, range(S.nch)))]
    _fire(l, S, Us, Vn, theta, c, st)


def _fire(l, S, Us, V, theta, c, st):
    """the last micro-batch: a code moves one level where |A| >= theta (within 0 .. levels), each move spends theta;
    row chunk by row chunk. The accumulator holds the evidence of g = dL/dW (the factors give no dL/dq = dL/dW * s:
    s varies along a row by group); a scale constant along a row cancels in u = g / sqrt(v) (v's row factor), so only
    s's variation within the row, rho = s / rms_row(s), is left, and it is applied here: A_q = rho A (a coarser group's
    move is worth more), and a move spends theta / rho of A (U -= theta (D / rho) V). rho is signed: s is trained and
    goes negative (15M from scratch: 56-78% of the groups by step 300), where a code moving up moves W down (with |s|
    the codes of those groups moved against their evidence: +0.14 to +0.17 at 300 steps)"""
    nf = torch.zeros((), device=V.device, dtype=torch.long)
    S.put_v(V)
    for cc, U in enumerate(Us):
        i, j = S.rng(cc)
        sc = l.s.detach()[i:j]           # signed: a group whose step went negative moves W down when its code moves up
        rho = (sc / sc.square().mean(1, keepdim=True).sqrt().clamp_min(1e-30)).repeat_interleave(l.g, 1); del sc
        A = (U @ V.T).mul_(rho); w0 = l.codes[i:j]
        d = A.sign(); nw = w0.float() + d
        fire = (A.abs() >= theta) & (nw >= 0) & (nw <= l.levels) & (rho.abs() > 1e-6); del A, nw
        w0.add_(torch.where(fire, d, torch.zeros_like(d)).to(torch.int8)); nf += fire.sum()
        D = torch.where(fire, d / rho, torch.zeros_like(d)); del d, rho, fire
        U = U - c["spend"] * (theta * (D @ V)); del D
        S.put_u(cc, U)
        Us[cc] = None
    S.flush()
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + l.N * l.K


class _EmbFn(torch.autograd.Function):
    """--intk-update factored: the embedding's lookup from the codes (only the batch's rows dequantized); the backward
    hands the TTF update its gradient as the distinct rows' summed dL/dh and returns the group parameters' gradients"""
    @staticmethod
    def forward(ctx, idx, lo, s, m):
        ctx.m = m; ctx.save_for_backward(idx)
        return m.deq_idx(idx.reshape(-1)).reshape(*idx.shape, m.K)

    @staticmethod
    def backward(ctx, go):
        (idx,) = ctx.saved_tensors; m = ctx.m
        u, inv = torch.unique(idx.reshape(-1), return_inverse=True)
        Gu = torch.zeros(u.numel(), m.K, device=go.device, dtype=torch.float32)
        Gu.index_add_(0, inv, go.reshape(-1, m.K).float())
        Gg = Gu.view(-1, m.K // m.g, m.g)
        q = m.codes[u].view(-1, m.K // m.g, m.g).float()
        f32 = dict(dtype=torch.float32, device=go.device)   # summed in fp32, returned in the parameters' dtype (bf16 at 27B)
        glo = torch.zeros(m.lo.shape, **f32).index_add_(0, u, Gg.sum(-1)).to(m.lo.dtype)
        gs = torch.zeros(m.s.shape, **f32).index_add_(0, u, (Gg * q).sum(-1)).to(m.s.dtype); del q, Gg
        m.add_part(("s", u, Gu))
        return None, glo, gs, None


def _logits(m, h, dt):
    """one token chunk's fp32 logits [n, V], the head dequantized in row chunks"""
    z = torch.empty(h.shape[0], m.N, dtype=torch.float32, device=h.device)
    for a in range(0, m.N, ROWS[0]):
        b = min(m.N, a + ROWS[0])
        z[:, a:b] = F.linear(h, m.deq_rows(a, b, dt)).float()
    return z


class _HeadLossFn(torch.autograd.Function):
    """--intk-update factored: the head and the (KD) loss of a micro-batch's final hidden states h [T, d] in token
    chunks, as headloss's frozen-head functions (dL/dh in the forward). The gradient goes to the update as its factors
    (the micro-batch's dL/dlogits [T, V] bf16 and h x grad_out) or, when that is the larger (T > 2 d: many tokens per
    micro-batch, a narrow model), as the dense gradient [V, d] fp32 summed over the token chunks. The group parameters'
    gradients are taken in the backward from either. kd: None | ("lp", log-probs [T, V], a) | ("topk", ids, log-probs,
    a, mode)"""
    @staticmethod
    def forward(ctx, h, lo, s, m, tgt, kd, n):
        from .headloss import ce_kd_dz, topk_dz
        from .update import _mmb
        T, dt = h.shape[0], h.dtype
        dense = T > 2 * m.K if DENSE_G[0] is None else DENSE_G[0]   # (bf16 T x V against fp32 V x d)
        gy = None if dense else torch.empty(T, m.N, dtype=torch.bfloat16, device=h.device)
        G = torch.zeros(m.N, m.K, dtype=torch.float32, device=h.device) if dense else None
        dh = torch.empty(T, m.K, dtype=torch.float32, device=h.device)
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        hb = h.detach().to(torch.bfloat16)
        for i in range(0, T, n):
            hc, tc = h[i:i + n], tgt[i:i + n]
            if kd is None:
                loss, dz = ce_kd_dz(_logits(m, hc, dt), tc, None, 0.0)
            elif kd[0] == "lp":
                loss, dz = ce_kd_dz(_logits(m, hc, dt), tc, kd[1][i:i + n], kd[2])
            else:
                loss, dz = topk_dz(_logits(m, hc, dt), tc, kd[1][i:i + n], kd[2][i:i + n], kd[3], kd[4])
            total += loss
            if dense:
                dzb = dz.to(torch.bfloat16); del dz
                G += _mmb(dzb.T, hb[i:i + n])
            else:
                gy[i:i + n] = dz; del dz
                dzb = gy[i:i + n]
            acc = torch.zeros(hc.shape[0], m.K, dtype=torch.float32, device=h.device)
            for a in range(0, m.N, ROWS[0]):
                b = min(m.N, a + ROWS[0])
                acc += _mmb(dzb[:, a:b], m.deq_rows(a, b))
            dh[i:i + n] = acc; del acc, dzb
        ctx.m, ctx.gy, ctx.G, ctx.dh, ctx.h, ctx.dt = m, gy, G, dh, (None if dense else hb), dt
        return total

    @staticmethod
    def backward(ctx, go):
        from .update import _mmb
        m, gy, G, dh, h = ctx.m, ctx.gy, ctx.G, ctx.dh, ctx.h
        ctx.gy = ctx.G = ctx.dh = ctx.h = None
        glo, gs = torch.empty_like(m.lo), torch.empty_like(m.s)
        Gn = m.K // m.g
        if G is not None:
            G.mul_(go.float())
        else:
            x = (h.float() * go.float()).to(torch.bfloat16)       # g = gy^T x carries grad_out (1 / tokens / accum)
        for a in range(0, m.N, ROWS[0]):
            b = min(m.N, a + ROWS[0])
            Gg = (G[a:b] if G is not None else _mmb(gy[:, a:b].T, x)).view(b - a, Gn, m.g)
            glo[a:b] = Gg.sum(-1)
            gs[a:b] = (Gg * m.codes[a:b].view(b - a, Gn, m.g).float()).sum(-1); del Gg
        m.add_part(("g", G, None) if G is not None else ("d", gy, x))
        return (dh * go.float()).to(ctx.dt), glo, gs, None, None, None, None


def head_loss(m, h, tgt, kd=None, chunk=512):
    """the summed loss of the int-k head m over the tokens of h [T, d] (training; see _HeadLossFn)"""
    n = max(64, min(chunk, TOK_BYTES[0] // (4 * m.N)) // 64 * 64)
    return _HeadLossFn.apply(h, m.lo, m.s, m, tgt, kd, n)


class _Emb(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @property
    def weight(self):
        return self.m.leaf()

    def forward(self, idx):
        if INTK_FACTORED[0] and torch.is_grad_enabled() and self.training:
            return _EmbFn.apply(idx, self.m.lo, self.m.s, self.m)
        if not torch.is_grad_enabled():
            return self.m.deq_idx(idx.reshape(-1)).reshape(*idx.shape, self.m.K)
        return F.embedding(idx, self.m.leaf())


class _Head(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @property
    def weight(self):
        return self.m.leaf()

    def forward(self, x):
        return F.linear(x, self.m.leaf().to(x.dtype))


def factored_head(mod, training):
    """the int-k matrix whose factored head loss replaces the dense head (--intk-update factored, training), or None"""
    m = getattr(mod, "m", None)
    return m if (isinstance(m, IntKMatrix) and INTK_FACTORED[0] and training and torch.is_grad_enabled()) else None


def enable_intk_emb(model, spec, log=print):
    """replace hf.model.embed_tokens / hf.lm_head by int-k code matrices (spec 'E:H[:G]'); returns the two matrices"""
    p = [int(v) for v in spec.split(":")]
    eb, hb, g = p[0], p[1], (p[2] if len(p) > 2 else 64)
    hf = model.hf
    E, H = hf.model.embed_tokens.weight, hf.lm_head.weight
    assert E is not H, "--intk-emb: an untied head (ttfconv.fold unties it)"
    me, mh = IntKMatrix(E.data, eb, g).to(E.device), IntKMatrix(H.data, hb, g).to(H.device)
    for i, m in enumerate((me, mh)):
        m._ts_idx = 20_000 + i
    hf.model.embed_tokens = _Emb(me); hf.lm_head = _Head(mh)
    err = [float((m.dequant() - W.float()).norm() / W.float().norm()) if not W.is_meta else float("nan")   # (--pipe:
           for m, W in ((me, E.data), (mh, H.data))]                                                         # another stage's)
    log(f"--intk-emb {spec}: embedding {eb}-bit, head {hb}-bit codes in groups of {g} (rel err {err[0]:.4f} / "
        f"{err[1]:.4f}), trained by TTF ({'factored' if INTK_FACTORED[0] else 'dense'} update); "
        f"{(me.lo.numel() + me.s.numel() + mh.lo.numel() + mh.s.numel()) / 1e6:.1f}M group parameters in the float tail")
    return me, mh


def enable_intk_emb_bt(model, spec, log=print):
    """the same for the trainer's own model (BitTransformer, training from scratch): tok_emb (and lm_head) as int-k
    codes from their random init; tied (no lm_head): one matrix of E bits for both, its update taking both gradients"""
    p = [int(v) for v in spec.split(":")]
    eb, hb, g = p[0], p[1], (p[2] if len(p) > 2 else 64)
    E = model.tok_emb.weight
    me = IntKMatrix(E.data, eb, g).to(E.device); me._ts_idx = 20_000
    out = [me]
    if model.lm_head is None:
        me.n_parts = 2
    else:
        H = model.lm_head.weight
        mh = IntKMatrix(H.data, hb, g).to(H.device); mh._ts_idx = 20_001
        model.lm_head = _Head(mh); out.append(mh)
    model.tok_emb = _Emb(me)
    log(f"--intk-emb {spec}: {'tied embedding / head' if len(out) == 1 else 'embedding / head'} as int-k codes "
        f"({eb}{'' if len(out) == 1 else f' / {hb}'}-bit, groups of {g}), trained by TTF "
        f"({'factored' if INTK_FACTORED[0] else 'dense'} update); "
        f"{sum(m.lo.numel() + m.s.numel() for m in out) / 1e6:.2f}M group parameters in the float tail")
    return out


def intk_layers(model):
    return [m for m in model.modules() if isinstance(m, IntKMatrix)]
