"""Model + training configuration for the ternary BitNet b1.58 LLM."""
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 1024               # hidden size
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 16          # set < n_heads for grouped-query attention (GQA)
    hidden_dim: int = 2816        # SwiGLU inner dim (~8/3*dim, rounded to multiple_of)
    multiple_of: int = 256
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True    # share token embedding with lm_head
    act_bits: int = 8              # activation quantization bit-width

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide n_kv_heads"

    # ---- parameter / memory accounting -------------------------------------
    def n_params(self) -> int:
        d, L, V, H = self.dim, self.n_layers, self.vocab_size, self.hidden_dim
        per_layer = 4 * d * d + 3 * d * H          # attn (q,k,v,o) + swiglu (gate,up,down)
        embed = V * d
        head = 0 if self.tie_embeddings else V * d
        norms = L * 2 * d + d                       # small; rope has none
        # BitLinear carries an internal RMSNorm weight per linear:
        bitlinear_norms = L * (4 * d + 2 * H + d)   # q,k,v,o inputs + gate,up + down input
        return per_layer * L + embed + head + norms + bitlinear_norms

    def n_ternary_params(self) -> int:
        """Weights stored ternary at deploy (the BitLinear weight matrices only)."""
        d, L, H = self.dim, self.n_layers, self.hidden_dim
        return (4 * d * d + 3 * d * H) * L

    def deploy_bytes(self) -> int:
        """Packed-ternary weights (~1.58 bit) + full-precision embeddings/norms (bf16)."""
        ternary_bits = self.n_ternary_params() * 1.58
        fp = (self.n_params() - self.n_ternary_params()) * 2  # bf16 bytes
        return int(ternary_bits / 8 + fp)

    def report(self) -> str:
        gib = 1024 ** 3
        return (
            f"params        : {self.n_params()/1e6:,.1f} M\n"
            f"  ternary      : {self.n_ternary_params()/1e6:,.1f} M (packed ~1.58 bit)\n"
            f"deploy size    : {self.deploy_bytes()/gib:.3f} GiB\n"
            f"train est bf16 : ~{self.n_params()*6/gib:.2f} GiB weights+8bit-Adam "
            f"(+ activations)"
        )


@dataclass
class TrainConfig:
    data_path: str = "data/train.bin"      # flat uint16 token stream (nanoGPT-style)
    val_path: str = "data/val.bin"
    out_dir: str = "checkpoints"
    batch_size: int = 4                     # micro-batch (per step)
    grad_accum: int = 16                    # effective batch = batch_size*grad_accum
    seq_len: int = 2048
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_checkpoint: bool = True            # trade compute for VRAM
    use_8bit_adam: bool = True              # needs bitsandbytes; else falls back to AdamW
    compile: bool = True
    eval_interval: int = 1000
    eval_iters: int = 100
    log_interval: int = 10
    save_interval: int = 2000
    seed: int = 1337
    dtype: str = "bfloat16"


# --- presets sized for a single consumer GPU -------------------------------
PRESETS = {
    # ~340M params. Trains comfortably on 12GiB with checkpointing.
    "d1024_l24": ModelConfig(dim=1024, n_layers=24, n_heads=16, n_kv_heads=16,
                             hidden_dim=2816),
    # ~130M params. Fast to iterate / debug the pipeline.
    "small": ModelConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=12,
                         hidden_dim=2048),
    # ~51M, GPT-2 BPE vocab (50257). For the master vs master-free study on Wiki.
    "s50": ModelConfig(vocab_size=50257, dim=512, n_layers=8, n_heads=8,
                       n_kv_heads=8, hidden_dim=1408, max_seq_len=512),
    # 2-layer toy for the induction-head test on synthetic repeated-segment data (scripts/toy/).
    "toy2": ModelConfig(vocab_size=2048, dim=256, n_layers=2, n_heads=4, n_kv_heads=4,
                        hidden_dim=768, max_seq_len=256),
    # ~730M params. Pushes a 12GiB card: use batch_size=1, grad_checkpoint, 8bit-adam.
    # ~1.3B: the 340M shape at twice the width (a third width for the rank-vs-width question).
    "d2048_l24": ModelConfig(dim=2048, n_layers=24, n_heads=16, n_kv_heads=16,
                             hidden_dim=5632),
    "d1536_l24": ModelConfig(dim=1536, n_layers=24, n_heads=16, n_kv_heads=8,
                             hidden_dim=4096),
    # ~27B. Deploys ~5.3GiB packed ternary. VERIFIED trainable on a 12GiB laptop
    # GPU (stateless mode, seq<=512, peak ~10.9GiB, ~67 tok/s). 32k vocab keeps
    # the embedding tail small; raise it if your tokenizer needs more.
    "b27": ModelConfig(vocab_size=32000, dim=5120, n_layers=83, n_heads=40,
                       n_kv_heads=8, hidden_dim=13824, max_seq_len=2048),
    # width sweep (3 Oct): ~10B and ~15B with the b27 head layout (GQA, 8 KV heads of 128)
    "b10": ModelConfig(vocab_size=32000, dim=5120, n_layers=36, n_heads=40,
                       n_kv_heads=8, hidden_dim=13824, max_seq_len=2048),
    "b15": ModelConfig(vocab_size=32000, dim=6144, n_layers=38, n_heads=48,
                       n_kv_heads=8, hidden_dim=16384, max_seq_len=2048),
    # ~2.9B (5 Oct, memory-matched goldbug arm): width 3072, the b27 head layout (GQA, 8 KV heads of 128)
    "b3": ModelConfig(vocab_size=32000, dim=3072, n_layers=28, n_heads=24,
                      n_kv_heads=8, hidden_dim=8192, max_seq_len=2048),
    # ~40B target. NOT trainable on one 12GiB card (weights alone ~8GiB ternary +
    # activations). Needs multi-GPU / big-RAM offload. Architecture is identical.
    "b40": ModelConfig(vocab_size=128000, dim=6144, n_layers=80, n_heads=48,
                       n_kv_heads=8, hidden_dim=16384, max_seq_len=4096),
}
DEFAULT_PRESET = "d1024_l24"
