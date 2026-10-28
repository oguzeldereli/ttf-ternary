"""Qwen3.5 / Qwen3.8 (model_type qwen3_5: Gated DeltaNet + gated full attention) with ternary linears, for --arch-hf.

The Hugging Face implementation (transformers >= 5.8, fla kernels for the delta rule) runs the model; every large
linear of the text decoder becomes a KernelTernaryLinear, so TTF's update, the row / column scales and the low-rank
adapter work unchanged. Kept in float: the token embedding and the output head (frozen, bf16: 248k x d), the RMSNorm
gains, the DeltaNet's short convolution, decay and gate parameters (A_log, dt_bias, in_proj_a, in_proj_b: 16-row
projections), the q / k norms. Text only: the vision tower and the multi-token-prediction layer are not built.

The wrapper takes the trainer's interface: forward(idx, targets) -> (None, mean loss), with the head and the
cross-entropy in chunks of loss_chunk tokens (the logits of a 248k vocabulary are never all materialised)."""
import contextlib
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# linears kept in float (tiny, or gates whose rounding would be all error)
_KEEP_FLOAT = ("in_proj_a", "in_proj_b")
FUSED_NORM = [False]     # (set by enable_fused_norm, which the trainer always calls) Qwen3_5RMSNorm through kernel.rms_norm_qwen (one kernel forward, one backward)


def enable_fused_norm():
    """route every Qwen3_5RMSNorm (not the gated one) through the fused kernel when the input is on the GPU"""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
    if getattr(Q.Qwen3_5RMSNorm, "_tern_fused", False):
        return
    orig = Q.Qwen3_5RMSNorm.forward

    def fwd(self, x):
        if x.is_cuda and FUSED_NORM[0]:
            from .kernel import rms_norm_qwen
            return rms_norm_qwen(x, self.weight, self.eps)
        return orig(self, x)
    Q.Qwen3_5RMSNorm.forward = fwd
    Q.Qwen3_5RMSNorm._tern_fused = True
    FUSED_NORM[0] = True
    morig = Q.Qwen3_5MLP.forward

    def mlp(self, x):     # silu(gate) * up in one kernel (bf16 on the GPU, silu MLPs only)
        if x.is_cuda and FUSED_NORM[0] and self.config.hidden_act == "silu":
            from .kernel import swiglu
            g, u = self.gate_proj(x), self.up_proj(x)
            if g.dtype == torch.bfloat16 and u.dtype == torch.bfloat16:
                return self.down_proj(swiglu(g, u))
            return self.down_proj(self.act_fn(g) * u)
        return morig(self, x)
    Q.Qwen3_5MLP.forward = mlp
    gorig = Q.Qwen3_5RMSNormGated.forward

    def gnorm(self, hidden_states, gate):    # the DeltaNet output's gated norm in one kernel (bf16 inputs on the GPU)
        if (hidden_states.is_cuda and FUSED_NORM[0] and hidden_states.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16
                and self.activation == "silu" and self.weight.dtype != torch.bfloat16):   # bf16 gains: not reproduced
            from .kernel import gated_norm_qwen
            return gated_norm_qwen(hidden_states, gate, self.weight, self.variance_epsilon)
        return gorig(self, hidden_states, gate)
    Q.Qwen3_5RMSNormGated.forward = gnorm


@contextlib.contextmanager
def _no_packed_check():
    """HF's causal-mask builder looks for packed sequences in position_ids and ends with `if (...).all()`: a host sync
    on every forward (~75 ms on a 0.8B adapter micro-batch, the GPU queue drained each time). Our windows are never
    packed (the model builds position_ids itself, an arange), so the check's answer is None: it is skipped while the
    model runs"""
    from transformers import masking_utils as mu
    f = mu.find_packed_sequence_indices
    mu.find_packed_sequence_indices = lambda position_ids: None
    try:
        yield
    finally:
        mu.find_packed_sequence_indices = f


def is_qwen35(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    return c.get("model_type", "").startswith("qwen3_5") or c.get("text_config", {}).get("model_type", "").startswith("qwen3_5")


def text_config(path):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(path)
    return c.text_config if hasattr(c, "text_config") else c


def model_config(path, max_seq_len=2048):
    """our ModelConfig with the text decoder's sizes (for the trainer's report and bookkeeping only)"""
    from .config import ModelConfig
    t = text_config(path)
    nh = t.num_attention_heads        # Qwen3.8-27B: 24 heads x head_dim 256 != hidden 5120, which ModelConfig rejects;
    if t.hidden_size % nh:            # its heads are unused here (transformers' attention runs), so report the KV heads
        nh = t.num_key_value_heads
    return ModelConfig(vocab_size=t.vocab_size, dim=t.hidden_size, n_layers=t.num_hidden_layers,
                       n_heads=nh, n_kv_heads=t.num_key_value_heads,
                       hidden_dim=t.intermediate_size, max_seq_len=max_seq_len,
                       tie_embeddings=bool(getattr(t, "tie_word_embeddings", False)))


class QwenTern(nn.Module):
    loss_chunk = 2048
    head_fused = True       # the loss and dL/dh in one pass where it applies (ttf/headloss.py; False: the autograd path,
    # the reference for tests)

    def __init__(self, hf_model, grad_checkpoint=True):
        super().__init__()
        self.hf = hf_model
        self.layers = hf_model.model.layers
        self.grad_checkpoint = grad_checkpoint
        self.ckpt_skip = 0
        self.kd_logp = None
        self.kd_w = 1.0
        self.kd_mode = "renorm"
        self.kd_topk = None      # --kd-cache: (ids [T, k] long, log-probs [T, k]) of the teacher's top k per token
        self.pp = None           # --pipe: {"rank", "n", "lo", "hi"} (see _stage)

    def _head_w(self):
        lm = self.hf.lm_head
        return lm.weight

    def _ckpt_setup(self):
        m = self.hf.model
        ckpt = self.grad_checkpoint and self.training
        if ckpt and not m.gradient_checkpointing:
            self.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self._skip_set = False
        elif not ckpt and m.gradient_checkpointing:
            self.hf.gradient_checkpointing_disable()
        if ckpt and not getattr(self, "_skip_set", False):                     # --ckpt-skip N: the last N decoder
            n_ = len(m.layers) if self.pp is None else self.pp["hi"]             # layers (--pipe: of each stage) keep
            for i_, l_ in enumerate(m.layers):                                   # their activations (no recompute),
                l_.gradient_checkpointing = i_ < n_ - self.ckpt_skip             # memory permitting
            self._skip_set = True

    def forward(self, idx, targets=None):
        m = self.hf.model
        self._ckpt_setup()
        if self.pp is not None:
            return self._pp_forward(idx, targets)
        with _no_packed_check():
            h = m(input_ids=idx, use_cache=False).last_hidden_state
        if targets is None:
            return F.linear(h, self._head_w().to(h.dtype)), None
        return None, self._head_loss(h, targets)

    def _head_loss(self, h, targets):
        """the mean loss of the final hidden states: the head and the (KD) loss in chunks of loss_chunk tokens"""
        ckpt = self.grad_checkpoint and self.training
        from .intk_emb import factored_head, head_loss as _ik_loss
        ik = factored_head(self.hf.lm_head, self.training)
        if ik is not None:      # --intk-emb, factored update: the int-k head's chunked loss (no [V, d] matrix, no dense dL/dW)
            with torch.profiler.record_function("loss.head_kd"):
                hf_, tf = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
                if self.kd_topk is not None:
                    kd = ("topk", self.kd_topk[0], self.kd_topk[1], self.kd_w, self.kd_mode)
                else:
                    kd = ("lp", self.kd_logp, self.kd_w) if self.kd_logp is not None else None
                return _ik_loss(ik, hf_, tf, kd, self.loss_chunk) / (tf != -1).sum().clamp_min(1)
        w = self._head_w()
        _lrf = torch.profiler.record_function("loss.head_kd"); _lrf.__enter__()   # the 248k head + the (KD) loss
        hf_, tf = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        n = self.loss_chunk
        total = hf_.new_zeros((), dtype=torch.float32)
        kd = self.kd_logp
        wc = w.to(hf_.dtype)      # the head in the activations' dtype once per forward, not once per chunk (same values)
        for i in range(0, hf_.shape[0], n):
            hc, tc = hf_[i:i + n], tf[i:i + n]
            if self.head_fused and self.training and wc.requires_grad and self.kd_topk is None and self.pp is None:
                from .headloss import head_loss_train_chunk      # trained head: dL/dW accumulated in
                if i == 0:                                       # the forward, returned once (headloss.py)
                    _acc = {}
                total = total + head_loss_train_chunk(hc, wc, tc, kd[i:i + n] if kd is not None else None,
                                                      self.kd_w if kd is not None else 0.0, _acc)
                continue
            if self.head_fused and self.training and not wc.requires_grad:
                from .headloss import head_loss_chunk, head_loss_topk_chunk   # the loss and dL/dh in one
                if self.kd_topk is not None:                                   # pass (frozen head)
                    total = total + head_loss_topk_chunk(hc, wc, tc, self.kd_topk[0][i:i + n],
                                                         self.kd_topk[1][i:i + n], self.kd_w, self.kd_mode)
                else:
                    total = total + head_loss_chunk(hc, wc, tc, kd[i:i + n] if kd is not None else None,
                                                    self.kd_w if kd is not None else 0.0)
                continue
            if self.kd_topk is not None:
                f_ = _kd_topk_chunk
                args_ = (hc, wc, tc, self.kd_topk[0][i:i + n], self.kd_topk[1][i:i + n], self.kd_w,
                         self.kd_mode)
            elif kd is not None:
                from .model import BitTransformer
                f_ = BitTransformer._loss_chunk_kd
                args_ = (hc, wc, tc, kd[i:i + n], self.kd_w)
            else:
                f_ = _ce_chunk
                args_ = (hc, wc, tc)
            total = total + (checkpoint(f_, *args_, use_reentrant=False) if ckpt else f_(*args_))
        out = total / (tf != -1).sum().clamp_min(1)
        _lrf.__exit__(None, None, None)
        return out

    # ---- pipeline parallel (--pipe): stage r of n holds decoder layers [lo, hi), the embedding on the first stage, the
    # final norm and the head on the last; the other layers stay on the meta device (no memory), so module order, names,
    # checkpoint keys and the TTF seeds (_ts_idx) are those of the single-GPU model. Only the hidden states cross stages
    # (B x T x d per micro-batch); every stage's TTF update sees only its own layers' gradients, so nothing is reduced.
    def _stage(self, x):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
        m = self.hf.model; P = self.pp
        h = m.embed_tokens(x) if P["rank"] == 0 else x
        B, T = h.shape[:2]
        pos = torch.arange(T, device=h.device).view(1, 1, -1).expand(4, B, -1)      # as Qwen3_5TextModel.forward
        tpos, pos = pos[0], pos[1:]
        kw = {"config": m.config, "inputs_embeds": h, "attention_mask": None, "past_key_values": None, "position_ids": tpos}
        masks = {"full_attention": Q.create_causal_mask(**kw), "linear_attention": Q.create_recurrent_attention_mask(**kw)}
        pe = m.rotary_emb(h, pos)
        for i in range(P["lo"], P["hi"]):
            h = m.layers[i](h, position_embeddings=pe, attention_mask=masks[m.config.layer_types[i]], position_ids=tpos,
                            past_key_values=None, use_cache=False)
        if P["rank"] == P["n"] - 1:
            h = m.norm(h)
        return h

    def _h_meta(self, x):
        e = self.hf.model.embed_tokens
        from .intk_emb import _Emb
        # (an int-k embedding's .weight dequantizes the whole matrix: its lookup's dtype instead)
        dt = torch.bfloat16 if isinstance(e, _Emb) else e.weight.dtype
        return (x.shape[0], x.shape[1], self.hf.config.hidden_size), dt

    def _pp_forward(self, idx, targets):
        """a forward without gradients (evaluation): the hidden states go down the stages, the last computes the loss,
        every stage returns it"""
        assert targets is not None and not torch.is_grad_enabled(), "--pipe: forward() is for evaluation; training: pp_train"
        P = self.pp; r, n = P["rank"], P["n"]
        sh, dt = self._h_meta(idx)
        h = self._stage(idx if r == 0 else _pp_recv(sh, dt, r - 1, idx.device))
        if r < n - 1:
            _pp_send(h, r + 1); loss = torch.zeros((), device=idx.device)
        else:
            loss = self._head_loss(h, targets).float()
        return None, _pp_bcast(loss, n - 1)

    def pp_train(self, batches, pre_bwd, autocast):
        """one step's micro-batches through the pipeline, 1F1B: stage r runs n - r - 1 warm-up forwards, then one forward
        and one backward in turn, then the remaining backwards, so at most n - r micro-batches' activations are alive on
        it (the old schedule, all forwards then all backwards, kept all M). The backwards still run in the order 0 .. M-1,
        as the single-GPU loop (pre_bwd(i) before each, for the TTF accumulation flags), each scaled 1 / M. In the steady
        state a forward's send and the next backward's receive go as one group (_pp_sendrecv), and a backward's send and
        the next forward's receive likewise: NCCL runs a rank pair's sends and receives in order on one stream, so as
        separate ops stage r's send of micro-batch i + 1 waited behind stage r + 1's send of the gradient of i, and that
        behind stage r's receive of it (a deadlock once a message exceeds NCCL's buffer: 27B, 2 x L40S, 25 Oct). Returns
        (the last micro-batch's loss, the mean loss of all M) on every stage. batches: [(x, y, kd_topk or None)]"""
        from . import update as _U
        from .layers import _rf
        P = self.pp; r, n = P["rank"], P["n"]; M = len(batches)
        self._ckpt_setup()
        keep, pend = {}, []
        dev = batches[0][0].device
        last = torch.zeros((), device=dev)
        tot = torch.zeros((), device=dev)

        def fwd(i, hin):
            """forward of micro-batch i from the received hidden states hin (stage 0: None); returns what goes to the
            next stage (the last stage: None)"""
            x, y, kd = batches[i]
            with autocast():
                if hin is not None:
                    hin.requires_grad_()
                h = self._stage(x if r == 0 else hin)
                if r < n - 1:
                    keep[i] = (hin, h); return h.detach()
                self.kd_topk = kd
                keep[i] = (hin, self._head_loss(h, y)); self.kd_topk = None

        def bwd(i, g):
            """backward of micro-batch i with the received gradient g (the last stage: None); returns the gradient for
            the previous stage (stage 0: None). A stage after the first holds its layers' last-micro-batch updates
            (the step's basis turn, firing and spend) until that gradient is sent (handover): the previous stage
            waited for them otherwise (27B r1536, 2 stages: ~4.7 s of orthonormalisation per step in series)"""
            nonlocal last, tot
            hin, out = keep.pop(i)
            pre_bwd(i)
            if r > 0 and i == M - 1:
                _U.DEFER[0] = []
            with torch.profiler.record_function("model.bwd"):
                if r == n - 1:
                    (out / M).backward(); last = out.detach().float(); tot += last
                else:
                    out.backward(g)
            return hin.grad if r > 0 else None

        def recv_f(i):
            return _pp_recv(*self._h_meta(batches[i][0]), r - 1, dev) if r > 0 else None

        def g_meta(i):
            return keep[i][1].shape, keep[i][1].dtype

        nw = min(n - r - 1, M)
        for i in range(nw):                       # (warm-up: never the last stage)
            pend.append(_pp_isend(fwd(i, recv_f(i)), r + 1))
        hin = recv_f(nw) if nw < M else None
        fi, bi = nw, 0
        def handover():
            """the held updates, once the last input gradient is on its way"""
            d, _U.DEFER[0] = _U.DEFER[0], None
            for l_, g_, s_, c_ in d or ():
                with _rf("ttf.update"):
                    _U.ts_layer(l_, g_, s_, c_)

        while fi < M:
            h = fwd(fi, hin); fi += 1
            g = _pp_sendrecv(h, r + 1, *g_meta(bi), dev, pend) if r < n - 1 else None
            gin = bwd(bi, g); bi += 1
            hin = None
            if r > 0:
                if fi < M:
                    hin = _pp_sendrecv(gin, r - 1, *self._h_meta(batches[fi][0]), dev, pend)
                else:
                    pend.append(_pp_isend(gin, r - 1))
                if bi == M:
                    handover()
        while bi < M:
            g = _pp_recv(*g_meta(bi), r + 1, dev) if r < n - 1 else None
            gin = bwd(bi, g); bi += 1
            if r > 0:
                pend.append(_pp_isend(gin, r - 1))
                if bi == M:
                    handover()
        for w, _buf in pend:
            w.wait()
        both = _pp_bcast(torch.stack([last, tot / M]), n - 1)
        return both[0], both[1]

    def float_tail_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def _ce_chunk(h, w, tgt):
    return F.cross_entropy(F.linear(h, w).float(), tgt, ignore_index=-1, reduction="sum")


def _kd_topk_chunk(h, w, tgt, ids, tlp, a, mode="rest"):
    """distillation to the teacher's top k (log-probabilities under its full softmax), mixed with next-token
    cross-entropy: (1 - a) CE + a KD, summed over tokens. mode 'rest': cross-entropy to the k tokens plus one bucket for
    all other tokens (the teacher's remaining mass against the student's), the KL on that coarsened distribution;
    'renorm': cross-entropy to the top k renormalised over the k (drops the teacher's tail, so the student is pushed to
    zero mass outside the k; at 87% top-k mass this cost Qwen 0.8B 0.17)"""
    z = F.linear(h, w).float()
    lse = torch.logsumexp(z, -1, keepdim=True)        # no full log-softmax: the k gathered logits minus the logsumexp
    lq = z.gather(1, ids) - lse
    if mode == "renorm":
        kd = -(torch.softmax(tlp.float(), -1) * lq).sum()
    else:
        pk = tlp.float().exp()
        p_rest = (1 - pk.sum(-1)).clamp_min(0)
        lq_rest = torch.logsumexp(z.scatter(1, ids, float("-inf")), -1) - lse[:, 0]
        kd = -(pk * lq).sum() - (p_rest * lq_rest).sum()
    if a >= 1.0:
        return kd
    ok = tgt != -1
    ce = -(z.gather(1, tgt.clamp_min(0)[:, None]) - lse)[:, 0][ok].sum()
    return (1 - a) * ce + a * kd


def _pp_gloo():
    import torch.distributed as dist
    return dist.get_backend() == "gloo"


def _pp_send(t, dst):
    import torch.distributed as dist
    t = t.contiguous()
    dist.send(t.cpu() if _pp_gloo() else t, dst)     # gloo (several stages on one GPU, a test): host tensors


def _pp_isend(t, dst):
    """non-blocking send; returns (work, the buffer, kept alive until work.wait())"""
    import torch.distributed as dist
    t = t.contiguous()
    b = t.cpu() if _pp_gloo() else t
    return dist.isend(b, dst), b


def _pp_sendrecv(t, peer, shape, dtype, dev, pend):
    """send t to peer and receive a (shape, dtype) tensor from it as one group (NCCL: both progress together, see
    pp_train); the send's (work, buffer) goes to pend. gloo (a test on one GPU): a buffered isend, then the receive"""
    import torch.distributed as dist
    if _pp_gloo():
        pend.append(_pp_isend(t, peer)); return _pp_recv(shape, dtype, peer, dev)
    t = t.contiguous(); buf = torch.empty(shape, dtype=dtype, device=dev)
    ws = dist.batch_isend_irecv([dist.P2POp(dist.isend, t, peer), dist.P2POp(dist.irecv, buf, peer)])
    for w in ws:                 # (one coalesced work for the group under NCCL)
        w.wait()
    pend.append((ws[0], t))
    return buf


def _pp_recv(shape, dtype, src, dev):
    import torch.distributed as dist
    buf = torch.empty(shape, dtype=dtype, device="cpu" if _pp_gloo() else dev)
    dist.recv(buf, src)
    return buf.to(dev)


def _pp_bcast(t, src):
    import torch.distributed as dist
    b = t.detach().float().reshape(-1)
    b = b.cpu() if _pp_gloo() else b
    dist.broadcast(b, src)
    return b.to(t.device).reshape(t.shape)


def pp_split(n_layers, n_stages, spec=""):
    """layer ranges per stage: spec 'a,b,c,...' (layer counts, sum n_layers) or an even split (earlier stages one more)"""
    c = [int(v) for v in spec.split(",")] if spec else [n_layers // n_stages + (i < n_layers % n_stages) for i in range(n_stages)]
    assert len(c) == n_stages and sum(c) == n_layers, f"--pipe-split {spec}: {n_stages} counts summing to {n_layers}"
    lo = [sum(c[:i]) for i in range(n_stages)]
    return [(lo[i], lo[i] + c[i]) for i in range(n_stages)]


def build_qwen(path, mode, make_linear=None, grad_checkpoint=True, device="cuda", dtype=torch.bfloat16, own=None):
    """the text decoder on `device`: mode 'fp32' (all linears float, the reference) or 'kernel' (make_linear for the large
    linears); float parts in `dtype`. Weights are NOT loaded here (a ternary checkpoint, or the
    ttf-conversion package's conversion). own=(lo, hi, first, last) (--pipe): only
    decoder layers [lo, hi) (+ the embedding if first, the final norm and head if last) are allocated, the rest stay on
    the meta device with the same module tree."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    t = text_config(path)
    t._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = Qwen3_5ForCausalLM(t)
    shapes = {}
    if mode == "kernel":    # the large linears are dropped before memory is allocated (27B: 50 GB of bf16 never made)
        for name, mod in list(hf.model.layers.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and cname not in _KEEP_FLOAT:
                    shapes[(name, cname)] = (child.in_features, child.out_features)
                    setattr(mod, cname, nn.Identity())
    if own is None:
        hf = hf.to_empty(device=device)
    else:
        lo, hi, first, last = own
        for mod in ([hf.model.layers[i] for i in range(lo, hi)] + ([hf.model.embed_tokens] if first else [])
                    + ([hf.model.norm, hf.lm_head] if last else [])):
            mod.to_empty(device=device)
    for p in hf.parameters():
        p.data = p.data.to(dtype)
    # the ternary layers are built after to_empty (which would leave their own buffers uninitialised), in fp32
    swapped = {}
    for (name, cname), (i, o) in shapes.items():
        mod = hf.model.layers.get_submodule(name) if name else hf.model.layers
        if own is None or own[0] <= int(name.split(".")[0]) < own[1]:
            new = make_linear(i, o).to(device)
        else:
            with torch.device("meta"):      # --pipe: another stage's layer
                new = make_linear(i, o)
        setattr(mod, cname, new)
        swapped[f"model.layers.{name}.{cname}" if name else f"model.layers.{cname}"] = new
    # the rotary module is rebuilt: to_empty leaves its inv_freq buffer uninitialised (non-persistent, not in checkpoints)
    hf.model.rotary_emb = type(hf.model.rotary_emb)(t).to(device)
    m = QwenTern(hf, grad_checkpoint=grad_checkpoint)
    return m, swapped


def _safetensor_index(path):
    """{tensor name (normalised to 'model.'): file} for lazy per-layer loading"""
    import glob
    idx = os.path.join(path, "model.safetensors.index.json")
    files = {}
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        for k, f in wm.items():
            files[k.replace("model.language_model.", "model.")] = (k, os.path.join(path, f))
    else:
        from safetensors import safe_open
        for f in glob.glob(os.path.join(path, "*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    files[k.replace("model.language_model.", "model.")] = (k, f)
    return files


def _get(files, name, dev):
    from safetensors import safe_open
    k, f = files[name]
    with safe_open(f, "pt", device="cpu") as h:
        return h.get_tensor(k).to(dev)
