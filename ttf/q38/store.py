"""Qwen3.8-Flash-Next checkpoint I/O on one GPU with the model in host memory (S, 5 Nov 2026).

Store(path): the FP8 release read once into host memory: per decoder layer the routed experts as stacked FP8 bytes
(gu [E, 2I, d] = gate rows then up, dn [E, d, I]) with their 128 x 128 block scales, and the layer's other tensors (bf16);
store.fill(L, i, dev) puts layer i on the GPU (experts dequantised there, chunked). The PLE n-gram table is read on
demand from the safetensors shards (MmapTable: np.memmap at the tensors' header offsets, only the rows a batch needs;
the page cache holds what is hot, and the kernel may drop it, so it costs no fixed host memory).
"""
import json, os, struct, time
import numpy as np
import torch

from ttf.q38 import model as M

PFX = M.PFX


def _header(fn):
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


_NP = {"F8_E4M3": (np.uint8, torch.float8_e4m3fn), "BF16": (np.uint16, torch.bfloat16), "F16": (np.float16, torch.float16),
       "F32": (np.float32, torch.float32)}


class MmapTable:
    """the PLE table of one layer, split in parts shard_0..shard_{P-1} (rows concatenated in index order); __getitem__
    with a 1-D LongTensor of global rows returns [n, dim] in the stored dtype (CPU)"""

    def __init__(self, path, layer, parts):
        idx = M.safetensor_index(path)
        p = f"{PFX}layers.{layer}.ple.ple_embedding.ngram_embedding."
        self.maps, starts, n0 = [], [], 0
        for j in range(parts):
            name = f"{p}shard_{j}.weight"
            fn = idx[name]
            h, base = _header(fn)
            m = h[name]
            npd, self.tdt = _NP[m["dtype"]]
            rows, dim = m["shape"]
            a, b = m["data_offsets"]
            self.maps.append(np.memmap(fn, dtype=npd, mode="r", offset=base + a, shape=(rows, dim)))
            starts.append(n0); n0 += rows
        self.dim, self.rows = dim, n0
        self.starts = torch.tensor(starts + [n0])
        self.npd = npd

    def __getitem__(self, rows):
        rows = rows.cpu().long()
        sh = torch.searchsorted(self.starts, rows, right=True) - 1
        out = np.empty((rows.numel(), self.dim), dtype=self.npd)
        order = torch.argsort(sh)
        shs, rs = sh[order].numpy(), rows[order].numpy()
        st = self.starts.numpy()
        bounds = np.searchsorted(shs, np.arange(len(self.maps) + 1))
        tmp = np.empty_like(out)
        for j in range(len(self.maps)):
            a, b = bounds[j], bounds[j + 1]
            if b > a:
                tmp[a:b] = self.maps[j][rs[a:b] - st[j]]
        out[order.numpy()] = tmp
        return torch.from_numpy(out).view(self.tdt)


class Store:
    def __init__(self, path, cfg, layers=None, log=print, experts=True, table_path=None):
        """table_path: where the PLE tables are read on demand (default path; e.g. the slim copy, so that path's files
        can be deleted after close())"""
        self.path, self.cfg = path, cfg
        self.table_path = table_path or path
        self.rd = M.Reader(path)
        n = cfg.num_hidden_layers if layers is None else layers
        self.E, self.I, self.d = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
        self.layers = []
        t0 = time.time()
        for i in range(n):
            self.layers.append(self._read(i, experts))
            if i % 8 == 0 or i == n - 1:
                log(f"store: layer {i} in host memory ({time.time() - t0:.0f}s)")
        self.glob = {k: self.rd.get(k).clone() for k in ("lm_head.weight", PFX + "embed_tokens.weight")}
        for k in self.rd.idx:
            if k.startswith(PFX + "hyper_connection_mixer."):
                self.glob[k] = self.rd.get(k).clone()
        self.tables = {}

    def _read(self, i, experts):
        p = f"{PFX}layers.{i}."
        rest = {}
        for k in self.rd.idx:
            if k.startswith(p) and ".mlp.experts." not in k and ".ngram_embedding.shard_" not in k:
                rest[k[len(p):]] = self.rd.get(k).clone()     # (a private copy, not a view of the mmapped file)
        if not experts:
            return {"rest": rest}
        E, I, d = self.E, self.I, self.d
        q0 = f"{p}mlp.experts.0."
        fp8 = self.rd.has(q0 + "gate_proj.weight_scale_inv")
        dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
        gu = torch.empty(E, 2 * I, d, dtype=dt); dn = torch.empty(E, d, I, dtype=dt)
        if fp8:
            sgu = torch.empty(E, 2 * I // 128, d // 128); sdn = torch.empty(E, d // 128, I // 128)
        for e in range(E):
            q = f"{p}mlp.experts.{e}."
            gu[e, :I] = self.rd.get(q + "gate_proj.weight"); gu[e, I:] = self.rd.get(q + "up_proj.weight")
            dn[e] = self.rd.get(q + "down_proj.weight")
            if fp8:
                sgu[e, :I // 128] = self.rd.get(q + "gate_proj.weight_scale_inv")
                sgu[e, I // 128:] = self.rd.get(q + "up_proj.weight_scale_inv")
                sdn[e] = self.rd.get(q + "down_proj.weight_scale_inv")
        out = {"rest": rest, "gu": gu, "dn": dn}
        if fp8:
            out.update(sgu=sgu, sdn=sdn)
        return out

    @staticmethod
    def _deq(w, s, dev, dtype=torch.bfloat16, chunk=64):
        """[E, N, K] FP8 (+ [E, N/128, K/128] scales) or bf16, on the host -> dtype on dev"""
        E, N, K = w.shape
        out = torch.empty(E, N, K, dtype=dtype, device=dev)
        for a in range(0, E, chunk):
            b = min(a + chunk, E)
            x = w[a:b].to(dev, non_blocking=False)
            if s is None:
                out[a:b] = x.to(dtype)
            else:
                sc = s[a:b].to(dev).float()
                out[a:b] = (x.float().view(b - a, N // 128, 128, K // 128, 128) * sc[:, :, None, :, None]).view(
                    b - a, N, K).to(dtype)
        return out

    def expert_weights(self, i, dev, dtype=torch.bfloat16):
        S = self.layers[i]
        return self._deq(S["gu"], S.get("sgu"), dev, dtype), self._deq(S["dn"], S.get("sdn"), dev, dtype)

    @torch.no_grad()
    def fill(self, L, i, dev, dtype=torch.bfloat16, experts=True):
        """layer i's weights onto L (parameters replaced, on dev); the PLE table attached on demand"""
        S = self.layers[i]["rest"]
        for n, t in list(L.named_parameters()):
            if n.startswith("mlp.gu.") or n.startswith("mlp.dn."):
                continue
            M.set_param(L, n, S[n].to(dev, dtype).view(t.shape))
        if L.ple is not None:
            ng = L.ple.ple_embedding
            for mine, theirs in ((ng.mult, "layer_multipliers"), (ng.sizes, "ngram_heads_vocab_sizes"),
                                 (ng.offs, "ngram_heads_offsets")):
                k = "ple.ple_embedding." + theirs
                if k in S:
                    assert torch.equal(mine.cpu(), S[k].cpu().long()), f"PLE {theirs} differs from the checkpoint"
            if i not in self.tables:
                self.tables[i] = MmapTable(self.table_path, i, self.cfg.split_ngram_parts)
                assert self.tables[i].rows >= ng.total, (self.tables[i].rows, ng.total)
            ng.table = self.tables[i]
            sk = "ple.ple_embedding.ngram_embedding.weight_scale"
            ng.scale = S[sk].float() if sk in S else None
            for b in ("mult", "sizes", "offs"):
                setattr(ng, b, getattr(ng, b).to(dev))
        if experts:
            gu, dn = self.expert_weights(i, dev, dtype)
            M.set_param(L, "mlp.gu.weight", gu); M.set_param(L, "mlp.dn.weight", dn)

    def close(self):
        """drop the open checkpoint handles (everything needed is in host memory, the tables come from table_path)"""
        self.rd.h.clear()

    @torch.no_grad()
    def unfill(self, L):
        """back to meta (frees the GPU copies)"""
        for n, t in list(L.named_parameters()):
            M.set_param(L, n, torch.empty(t.shape, dtype=t.dtype, device="meta"))

    @torch.no_grad()
    def fill_globals(self, m, dev, dtype=torch.bfloat16):
        M.set_param(m, "embed_tokens.weight", self.glob[PFX + "embed_tokens.weight"].to(dtype))      # host (lookup)
        M.set_param(m, "lm_head.weight", self.glob["lm_head.weight"].to(dev, dtype))
        for n, t in list(m.hyper_connection_mixer.named_parameters()):
            M.set_param(m.hyper_connection_mixer, n, self.glob[PFX + "hyper_connection_mixer." + n].to(dev, dtype))
        m.rotary.to(dev)
