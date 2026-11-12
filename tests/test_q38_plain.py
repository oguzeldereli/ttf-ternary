"""ttf.q38.model's plain mode (qwen3_5_moe checkpoints: Ornith-1.5-35B-A3B / Tiel-Coder) vs transformers'
Qwen3_5MoeForCausalLM on a tiny random config (fp32, GPU: the gated delta net's kernels).
  python -m pytest -q tests/test_q38_plain.py"""
import pytest
import torch

Q = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402

pytestmark = pytest.mark.gpu

TEXT = dict(  # Ornith-1.5-35B-A3B's text_config, shrunk
    model_type="qwen3_5_moe_text", attention_bias=False, attention_dropout=0.0, attn_output_gate=True,
    full_attention_interval=4, hidden_act="silu", linear_conv_kernel_dim=4, mamba_ssm_dtype="float32",
    max_position_embeddings=4096, partial_rotary_factor=0.25, rms_norm_eps=1e-6, router_aux_loss_coef=0.0,
    tie_word_embeddings=False, use_cache=False, output_router_logits=False,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.25,
                     "rope_theta": 10000000, "rope_type": "default"},
    hidden_size=128, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=32, linear_value_head_dim=32, num_experts=8, num_experts_per_tok=2,
    moe_intermediate_size=64, shared_expert_intermediate_size=64, vocab_size=512,
    pad_token_id=0, bos_token_id=1, eos_token_id=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_plain_matches_transformers():
    from ttf.q38 import model as M
    torch.manual_seed(0)
    known = {k: v for k, v in TEXT.items() if k not in ("model_type", "dtype")}
    hcfg = Qwen3_5MoeTextConfig(**known)
    hcfg._attn_implementation = "eager"
    ref = Q.Qwen3_5MoeForCausalLM(hcfg).float().eval().cuda()
    for n, p in ref.named_parameters():
        p.data.normal_(0, 0.05)
        if "A_log" in n:
            p.data.uniform_(0, 1)
    sd = ref.state_dict()
    cfg = M.text_config({"model_type": "qwen3_5_moe", "text_config": dict(TEXT)})
    assert cfg.plain and cfg.hc_count == 1
    m = M.build(cfg, device="cpu").float().eval().cuda()
    for i, L in enumerate(m.layers):
        for n, p in list(L.named_parameters()):
            if n in ("mlp.gu.weight", "mlp.dn.weight"):
                continue
            k = f"model.layers.{i}.mlp.gate.weight" if n.startswith("mlp.gate.") else f"model.layers.{i}." + n
            p.data.copy_(sd[k].view(p.shape))
        L.mlp.gu.weight.data.copy_(sd[f"model.layers.{i}.mlp.experts.gate_up_proj"])
        L.mlp.dn.weight.data.copy_(sd[f"model.layers.{i}.mlp.experts.down_proj"])
    m.embed_tokens.weight.data.copy_(sd["model.embed_tokens.weight"])
    m.lm_head.weight.data.copy_(sd["lm_head.weight"])
    m.hyper_connection_mixer.weight.data.copy_(sd["model.norm.weight"])
    M.FUSED_COMBINE[0] = False
    ids = torch.randint(0, 512, (2, 24)).cuda()
    with torch.no_grad():
        a = ref(input_ids=ids).logits
        b = m(ids)
    err = (a - b).abs().max().item() / a.abs().max().item()
    assert err < 1e-4, err
