"""ttf.export.io: the .ttf container (GGUF v3): write and read.
Needs the `gguf` package (pip install gguf, or llama.cpp's gguf-py)."""
import json
import os

import numpy as np
import torch


def write_ttf(path, sd, derived, ir, family, config, D, tok_dir=None, meta=None, extra_kv=None):
    import gguf
    w = gguf.GGUFWriter(path, "ttf")
    w.add_uint32("ttf.format_version", 1)
    w.add_string("ttf.family", family)
    w.add_string("ttf.config", json.dumps(config))
    w.add_string("ttf.compute_dtype", D)
    w.add_string("ttf.layout.trits", "rows5")
    w.add_string("ttf.ckpt_meta", json.dumps(meta or {}))
    w.add_string("ttf.ir", json.dumps(ir))
    for k, v in (extra_kv or {}).items():
        w.add_string(k, v)
    if tok_dir:
        for f, key in (("tokenizer.json", "ttf.tokenizer.json"), ("tokenizer_config.json", "ttf.tokenizer.config")):
            fp = os.path.join(tok_dir, f)
            if os.path.exists(fp):
                w.add_string(key, open(fp).read())
    for name, t in list(sd.items()) + list(derived.items()):
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.uint8:          # GGUF has no U8: the bytes as I8 (the layout tag says how to read them)
            w.add_tensor(name, t.view(torch.int8).numpy(), raw_dtype=gguf.GGMLQuantizationType.I8)
        elif t.dtype == torch.bfloat16:
            w.add_tensor(name, t.view(torch.int16).numpy().view(np.uint16), raw_shape=tuple(t.shape),
                         raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(name, t.float().numpy() if t.is_floating_point() else t.numpy())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def read_ttf(path):
    """-> (kv dict of strings / ints, tensors {name: torch}) with the I8 trit bytes back as uint8"""
    import gguf
    r = gguf.GGUFReader(path)
    kv = {}
    for f in r.fields.values():
        if f.name.startswith("ttf."):
            v = f.parts[f.data[0]]
            kv[f.name] = bytes(v).decode() if f.types[0] == gguf.GGUFValueType.STRING else int(v[0])
    W = {}
    for t in r.tensors:
        a = np.array(t.data)
        shape = tuple(int(s) for s in reversed(t.shape))
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            W[t.name] = torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16).reshape(shape)
        elif t.name.endswith((".wpacked", ".codes")):      # stored I8, read back as the bytes they are
            W[t.name] = torch.from_numpy(a.view(np.uint8).copy()).reshape(shape)
        else:
            W[t.name] = torch.from_numpy(a.copy()).reshape(shape)
    return kv, W
