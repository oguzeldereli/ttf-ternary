"""Qwen3.5 / Qwen3.8 (model_type qwen3_5: Gated DeltaNet + gated full attention) with ternary linears, for --init-hf.

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
FUSED_NORM = [False]     # --fused-norm: Qwen3_5RMSNorm through kernel.rms_norm_qwen (one kernel forward, one backward)


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


def _text_state(path):
    """the language model's tensors with the 'model.language_model.' / 'model.' prefixes normalised to 'model.'"""
    from .hf_init import load_hf_tensors
    sd = load_hf_tensors(path)
    out = {}
    for k, v in sd.items():
        if k.startswith("model.visual") or k.startswith("mtp") or ".mtp" in k:
            continue
        k2 = k.replace("model.language_model.", "model.")
        out[k2] = v
    return out


class QwenTern(nn.Module):
    loss_chunk = 2048
    head_fused = False      # --head-fused (ttf/headloss.py)

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
        if ckpt and self.ckpt_skip and not getattr(self, "_skip_set", False):   # --ckpt-skip N: the last N decoder
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
                from .headloss import head_loss_train_chunk      # --head-fused, trained head: dL/dW accumulated in
                if i == 0:                                       # the forward, returned once (headloss.py)
                    _acc = {}
                total = total + head_loss_train_chunk(hc, wc, tc, kd[i:i + n] if kd is not None else None,
                                                      self.kd_w if kd is not None else 0.0, _acc)
                continue
            if self.head_fused and self.training and not wc.requires_grad:
                from .headloss import head_loss_chunk, head_loss_topk_chunk   # --head-fused: loss and dL/dh in one
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
        the last micro-batch's loss (what the loop logs) on every stage. batches: [(x, y, kd_topk or None)]"""
        P = self.pp; r, n = P["rank"], P["n"]; M = len(batches)
        self._ckpt_setup()
        keep, pend = {}, []
        dev = batches[0][0].device
        last = torch.zeros((), device=dev)

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
            the previous stage (stage 0: None)"""
            nonlocal last
            hin, out = keep.pop(i)
            pre_bwd(i)
            with torch.profiler.record_function("model.bwd"):
                if r == n - 1:
                    (out / M).backward(); last = out.detach().float()
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
        while bi < M:
            g = _pp_recv(*g_meta(bi), r + 1, dev) if r < n - 1 else None
            gin = bwd(bi, g); bi += 1
            if r > 0:
                pend.append(_pp_isend(gin, r - 1))
        for w, _buf in pend:
            w.wait()
        return _pp_bcast(last, n - 1)

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
    b = t.detach().float().reshape(1)
    b = b.cpu() if _pp_gloo() else b
    dist.broadcast(b, src)
    return b.to(t.device)[0]


def pp_split(n_layers, n_stages, spec=""):
    """layer ranges per stage: spec 'a,b,c,...' (layer counts, sum n_layers) or an even split (earlier stages one more)"""
    c = [int(v) for v in spec.split(",")] if spec else [n_layers // n_stages + (i < n_layers % n_stages) for i in range(n_stages)]
    assert len(c) == n_stages and sum(c) == n_layers, f"--pipe-split {spec}: {n_stages} counts summing to {n_layers}"
    lo = [sum(c[:i]) for i in range(n_stages)]
    return [(lo[i], lo[i] + c[i]) for i in range(n_stages)]


def _swap_linears(hf_model, make_linear):
    """replace the decoder's nn.Linear modules (except _KEEP_FLOAT) by make_linear(in, out); returns {name: module}"""
    swapped = {}
    for name, mod in list(hf_model.model.layers.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname not in _KEEP_FLOAT:
                assert child.bias is None, f"{name}.{cname} has a bias"
                new = make_linear(child.in_features, child.out_features)
                setattr(mod, cname, new)
                swapped[f"model.layers.{name}.{cname}" if name else f"model.layers.{cname}"] = new
    return swapped


def build_qwen(path, mode, make_linear=None, grad_checkpoint=True, device="cuda", dtype=torch.bfloat16, own=None):
    """the text decoder on `device`: mode 'fp32' (all linears float, the reference) or 'kernel' (make_linear for the large
    linears); float parts in `dtype`. Weights are NOT loaded here (load_qwen). own=(lo, hi, first, last) (--pipe): only
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


@torch.no_grad()
def load_qwen(m, path, swapped, method="als", rc=True, hess=None, log=print):
    """copy the float tensors; convert each swapped linear's weight to trits + row / column scales (hf_init.ternarize /
    ternarize_gptq); embedding and head are frozen"""
    from .hf_init import ternarize, ternarize_gptq, _rel_err, _out_err, GPTQ_OPTS, mag_svd_fit
    from .kernel import pack_rows, trit_beta
    kept = []
    sd = _text_state(path)
    hf = m.hf
    own = dict(hf.named_parameters())
    sw = {k + ".weight" for k in swapped}
    missing = []
    for k, p in own.items():
        if k in sd:
            p.copy_(sd[k].to(p.dtype))
        elif k == "lm_head.weight" and "model.embed_tokens.weight" in sd:
            p.copy_(sd["model.embed_tokens.weight"].to(p.dtype))
        elif not any(k.startswith(s.rsplit(".weight", 1)[0]) for s in sw):
            missing.append(k)
    if missing:
        log(f"load_qwen: {len(missing)} float tensors not in the checkpoint (kept at init): {missing[:6]}")
    hf.model.embed_tokens.weight.requires_grad_(False); hf.lm_head.weight.requires_grad_(False)
    errs, oerrs = [], []
    for name, lin in swapped.items():
        W = sd[name + ".weight"].to(lin.wpacked.device).float()
        T, a, c = ternarize(W, "als" if method in ("gptq", "gptqc") else method)
        if hess is not None:
            Hl = hess[name].to(W.device)
            mu = hess["mu:" + name].to(W.device) if method == "gptqc" else None   # (centred: the layer's error)
            e0 = _out_err(W, T, a, c, Hl, mu)
            if method == "gptq":
                T, a, c = ternarize_gptq(W, Hl, a, c)
            elif method == "gptqc":     # centred (hf_init gptqc): GPTQ on the input covariance, the layer takes the
                T, a, c = ternarize_gptq(W, Hl - torch.outer(mu, mu), a, c)   # input mean out before its activation
                lin.enable_centering(mu, W @ mu)                            # quantisation, W mu as a bias
            oerrs.append((e0, _out_err(W, T, a, c, Hl, mu))); del Hl
        errs.append(_rel_err(W, T, a, c))
        lin.wpacked.copy_(pack_rows(T))
        beta = float(trit_beta(lin.wpacked, lin.K)) if lin.use_beta else 1.0
        if rc and lin.row_scale is not None:
            lin.row_scale.copy_(a / beta); lin.col_scale.copy_(c)
        else:
            raise SystemExit("Qwen conversion needs --rc-scale")
        if GPTQ_OPTS["mag_svd"] and hess is not None and method in ("gptq", "gptqc"):   # --adapter-init svd
            Hc = hess[name].to(W.device)
            if method == "gptqc":
                mu = hess["mu:" + name].to(W.device); Hc = Hc - torch.outer(mu, mu)
            kept.append(mag_svd_fit(lin, W, T, a, c, beta, Hc, GPTQ_OPTS["mag_svd"])); del Hc
        del W
    if errs:
        log(f"--init-hf {path}: {method}, {len(errs)} ternary linears, relative weight error mean "
            f"{sum(errs) / len(errs):.4f} (max {max(errs):.4f})")
    if oerrs:
        e0, e1 = zip(*oerrs)
        log(f"  output-space error mean {sum(e1) / len(e1):.4f} (ALS rounding {sum(e0) / len(e0):.4f})")
    if kept:
        log(f"  --adapter-init svd {GPTQ_OPTS['mag_svd']}: the adapters take {sum(kept) / len(kept):.3f} of the output-space "
            f"rounding error (min {min(kept):.3f}, max {max(kept):.3f}) over {len(kept)} linears")


@torch.no_grad()
def calib_hessians_qwen(path, data_path, n_seq=64, seq=2048, dev="cuda", dtype_u32=True, log=print):
    """input second moments X^T X / tokens of every large linear in the full-precision model ({linear name: H})"""
    import numpy as np
    m, _ = build_qwen(path, "fp32", grad_checkpoint=False, device=dev)
    sd = _text_state(path)
    for k, p in m.hf.named_parameters():
        src = sd.get(k, sd.get("model.embed_tokens.weight") if k == "lm_head.weight" else None)
        if src is not None:
            p.data.copy_(src.to(p.dtype))
    del sd
    m.eval()
    H, S, cnt, hooks = {}, {}, {}, []
    for name, mod in m.hf.model.layers.named_modules():
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] not in _KEEP_FLOAT:
            key = "model.layers." + name
            H[key] = torch.zeros(mod.in_features, mod.in_features, device=dev); cnt[key] = 0
            S[key] = torch.zeros(mod.in_features, device=dev)

            def hook(mod_, inp, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[key].addmm_(x.T, x); S[key] += x.sum(0); cnt[key] += x.shape[0]
            hooks.append(mod.register_forward_pre_hook(hook))
    data = np.memmap(data_path, dtype=np.uint32 if dtype_u32 else np.uint16, mode="r")
    rng = np.random.default_rng(4321)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for s in rng.integers(0, len(data) - seq - 1, n_seq):
            x = torch.tensor(data[s:s + seq].astype(np.int64), device=dev)[None]
            m(x, x)
    for h in hooks:
        h.remove()
    out = {k: (v / max(cnt[k], 1)).cpu() for k, v in H.items()}
    out.update({"mu:" + k: (v / max(cnt[k], 1)).cpu() for k, v in S.items()})   # input means (gptqc)
    del m, H, S; torch.cuda.empty_cache()
    log(f"calibration: input second moments (and means) of {len(out) // 2} linears from {n_seq} x {seq} tokens")
    return out


class _StopFwd(Exception):
    pass


@torch.no_grad()
def layer_inputs(m, windows, mb, dev):
    """the first decoder layer's inputs for the windows (micro-batches of mb) and the keyword arguments transformers
    passes to EACH layer (rotary embeddings; the full-attention and the recurrent masks differ by layer type), taken
    from one full forward of the first micro-batch; the other micro-batches stop at layer 0"""
    import numpy as np
    layers = m.hf.model.layers
    KW, H, hooks = [None] * len(layers), [], []

    def mk(i):
        def pre(mod, args, kwargs):
            if KW[i] is None:
                KW[i] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            if i == 0:
                H.append((args[0] if args else kwargs["hidden_states"]).detach())
                if all(k is not None for k in KW):
                    raise _StopFwd
        return pre
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_pre_hook(mk(i), with_kwargs=True))
    for b in range(0, len(windows), mb):
        x = torch.tensor(np.stack([w.astype(np.int64) for w in windows[b:b + mb]]), device=dev)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                m.hf.model(input_ids=x, use_cache=False)
        except _StopFwd:
            pass
    for h in hooks:
        h.remove()
    return H, KW


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


@torch.no_grad()
def convert_qwen_streaming(m, path, swapped, data_path, method="gptq", n_seq=64, seq=2048, mb=4, dev="cuda",
                           dtype_u32=True, log=print):
    """sequential, layer-by-layer conversion for models that do not fit as a whole in full precision (27B): the float
    tensors are loaded first; then for each decoder layer in turn its full-precision linears are put in place (read
    lazily from the safetensors), the calibration activations (inputs as produced by the already converted layers below)
    are run through it to collect each linear's input second moment, the linears are rounded (GPTQ or ALS) into the
    ternary layers, and the activations go on through the converted layer. Peak: the model's ternary form + one layer
    in full precision + the calibration activations (n_seq x seq x d bf16)."""
    import numpy as np
    from .hf_init import ternarize, ternarize_gptq, _rel_err, _out_err, GPTQ_OPTS, mag_svd_fit
    from .kernel import pack_rows, trit_beta
    from . import ext as X_
    kept = []
    files = _safetensor_index(path)
    hf = m.hf
    wrapped = {n: q for n, q in swapped.items() if X_.base(q) is not q}   # --ext wrappers: their ternary layer is rounded and
    swapped = {n: X_.base(q) for n, q in swapped.items()}                 # carries the stream; the wrapper goes back at the end
    sw = {k + ".weight" for k in swapped}
    for k, p in hf.named_parameters():                  # the float parts
        src = k if k in files else ("model.embed_tokens.weight" if k == "lm_head.weight" else None)
        if src in files and k not in sw:
            p.copy_(_get(files, src, dev).to(p.dtype))
    hf.model.embed_tokens.weight.requires_grad_(False); hf.lm_head.weight.requires_grad_(False)
    for f_ in X_.having("begin"):
        f_(m, files, dev, log)
    _prop_w = any(getattr(e, "propagate_wrapped", False) for e in X_.active())
    data = np.memmap(data_path, dtype=np.uint32 if dtype_u32 else np.uint16, mode="r")
    rng = np.random.default_rng(4321)
    starts = rng.integers(0, len(data) - seq - 1, n_seq)
    layers = hf.model.layers
    H_in, KW = layer_inputs(m, [data[s:s + seq] for s in starts], mb, dev)
    log(f"streaming conversion: {len(layers)} layers, {n_seq} x {seq} calibration tokens in {len(H_in)} micro-batches")
    errs, oerrs = [], []
    for li, layer in enumerate(layers):
        pre_ = f"model.layers.{li}."
        mine = {n: l for n, l in swapped.items() if n.startswith(pre_)}
        fp = {}                                         # this layer's full-precision linears, in place of the ternary ones
        for n, lin in mine.items():
            W = _get(files, n + ".weight", dev)
            for f_ in X_.having("weight"):              # e.g. a residual-stream fold (rounded to bf16 with the rest)
                W = f_(n, W.float(), li)
            f = nn.Linear(W.shape[1], W.shape[0], bias=False, device=dev, dtype=torch.bfloat16); f.weight.copy_(W)
            parent = layer.get_submodule(n[len(pre_):].rsplit(".", 1)[0]); setattr(parent, n.rsplit(".", 1)[1], f)
            fp[n] = (f, parent)
        Hs = {n: torch.zeros(f.in_features, f.in_features, device=dev) for n, (f, _) in fp.items()}
        Ss = {n: torch.zeros(f.in_features, device=dev) for n, (f, _) in fp.items()}
        cnt = {n: 0 for n in fp}
        hooks = []
        for n, (f, _) in fp.items():
            def hook(mod_, inp, n=n):
                xx = inp[0].reshape(-1, inp[0].shape[-1]).float(); Hs[n].addmm_(xx.T, xx); Ss[n] += xx.sum(0)
                cnt[n] += xx.shape[0]
            hooks.append(f.register_forward_pre_hook(hook))
        kw = KW[li]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for h in H_in:
                layer(h, **kw)
        for hh in hooks:
            hh.remove()
        for n, (f, parent) in fp.items():               # round into the ternary layer, put it back
            lin = mine[n]
            W = f.weight.float(); Hl = Hs.pop(n) / max(cnt[n], 1); mu = Ss.pop(n) / max(cnt[n], 1)
            T, a, c = ternarize(W, "als")
            mu_e = mu if method == "gptqc" else None    # (centred: the layer's error, not the raw input's)
            e0 = _out_err(W, T, a, c, Hl, mu_e)
            if method == "gptq":
                T, a, c = ternarize_gptq(W, Hl, a, c)
            elif method == "gptqc":                     # centred (hf_init gptqc)
                Hc_ = Hl - torch.outer(mu, mu)
                T, a, c = ternarize_gptq(W, Hc_, a, c)
                for f_ in X_.having("refine"):          # e.g. coordinate descent on the trits
                    T, a, c = f_(n, W, Hc_, T, a, c)
                lin.enable_centering(mu, W @ mu)
            oerrs.append((e0, _out_err(W, T, a, c, Hl, mu_e))); errs.append(_rel_err(W, T, a, c))
            lin.wpacked.copy_(pack_rows(T))
            beta = float(trit_beta(lin.wpacked, lin.K)) if lin.use_beta else 1.0
            lin.row_scale.copy_(a / beta); lin.col_scale.copy_(c)
            if method == "gptqc":
                for f_ in X_.having("fitted"):          # e.g. adapters fitted to what the trits left (the layer written)
                    f_(n, wrapped.get(n, lin), W, Hc_, mu, T, a, c)
                del Hc_
            if GPTQ_OPTS["mag_svd"] and method in ("gptq", "gptqc"):          # --adapter-init svd
                kept.append(mag_svd_fit(lin, W, T, a, c, beta, Hl - torch.outer(mu, mu) if method == "gptqc" else Hl,
                                        GPTQ_OPTS["mag_svd"]))
            # the calibration stream goes on through the bare ternary layer, unless a plug-in asks for the whole wrapper
            # (ext attribute propagate_wrapped: the layers above are then rounded for the inputs they will see)
            setattr(parent, n.rsplit(".", 1)[1], wrapped[n] if n in wrapped and _prop_w else lin)
            del W, Hl, T
        del fp, Hs, Ss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # on through the converted layer
            H_in = [(lambda o: o[0] if isinstance(o, tuple) else o)(layer(h, **kw)) for h in H_in]
        torch.cuda.empty_cache()
        if li % 8 == 0 or li == len(layers) - 1:
            e0, e1 = zip(*oerrs[-len(mine):])
            per = " ".join(f"{n.rsplit('.', 2)[-2]}.{n.rsplit('.', 1)[-1]} {e:.3f}" for n, e in zip(mine, e1))
            log(f"  layer {li}: {len(mine)} linears, output-space error {sum(e1) / len(e1):.4f} "
                f"(ALS {sum(e0) / len(e0):.4f}), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; {per}")
    for n, q in wrapped.items():                         # the wrappers back in the tree
        par, cn = n[len("model.layers."):].rsplit(".", 1)
        setattr(hf.model.layers.get_submodule(par), cn, q)
    for f_ in X_.having("end"):
        f_(m, log)
    e0, e1 = zip(*oerrs)
    if kept:
        log(f"--adapter-init svd {GPTQ_OPTS['mag_svd']}: the adapters take {sum(kept) / len(kept):.3f} of the output-space "
            f"rounding error (min {min(kept):.3f}, max {max(kept):.3f}) over {len(kept)} linears")
    log(f"--init-hf {path}: {method} (sequential, streaming), {len(errs)} ternary linears, weight error "
        f"{sum(errs) / len(errs):.4f}, output-space error {sum(e1) / len(e1):.4f} (ALS {sum(e0) / len(e0):.4f})")
