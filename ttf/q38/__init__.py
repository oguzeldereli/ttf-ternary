"""ttf.q38: Qwen3.8 (hybrid DeltaNet / attention, MoE, PLE n-gram embedding) in TTF: the model (model), the batched
expert TTF layers and update (moe), the Triton grouped GEMMs (moe_gg), the layer-major trainer (train), the activation /
teacher store (store) and the compressed PLE table (ple_pq). The model needs transformers >= 5.19 (Qwen3.5-MoE)."""
