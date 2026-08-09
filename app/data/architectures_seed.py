"""Open-weight model architectures, for VRAM estimation.

Not in the database, and deliberately so. Every other catalog table exists
because its contents go stale: prices move, tools get deprecated, and the
provenance machinery is there to say how old a figure is. A model's layer
count is fixed at publication and cannot drift - Llama 3.1 8B will have 32
layers forever. Giving it a `last_verified_at` would imply a freshness
question that does not exist, and a staleness chip on an immutable fact
teaches people to ignore the chips that matter.

So this is a module, served straight from memory, with no migration and no
seed step.

`kv_heads` is the field that matters most. Grouped-query attention shares key
and value projections across query heads, and the KV cache scales with
`kv_heads`, not `heads`. Llama 3.1 8B has 32 query heads and 8 KV heads, so
assuming multi-head attention overstates its cache by exactly 4x - on the
models people actually self-host, which is the whole point of the tool.

`head_dim` is stored rather than derived. It is usually `hidden / heads`, and
on Gemma 2 9B and Llama 3.2 1B it is not.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class Architecture(NamedTuple):
    key: str
    name: str
    family: str
    params_b: float
    layers: int
    hidden_size: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate_size: int
    max_context: int
    """Published maximum, which is often larger than what fits in one GPU."""

    @property
    def uses_gqa(self) -> bool:
        return self.kv_heads < self.heads

    @property
    def params(self) -> int:
        return int(self.params_b * 1_000_000_000)


ARCHITECTURES: Final[tuple[Architecture, ...]] = (
    # ---- Meta Llama -------------------------------------------------------
    Architecture(
        "llama-3.1-8b", "Llama 3.1 8B", "llama", 8.03, 32, 4096, 32, 8, 128, 14336, 131_072
    ),
    Architecture(
        "llama-3.3-70b", "Llama 3.3 70B", "llama", 70.6, 80, 8192, 64, 8, 128, 28672, 131_072
    ),
    Architecture(
        "llama-3.2-3b", "Llama 3.2 3B", "llama", 3.21, 28, 3072, 24, 8, 128, 8192, 131_072
    ),
    # 1B is the odd one: head_dim is 64, not hidden/heads.
    Architecture("llama-3.2-1b", "Llama 3.2 1B", "llama", 1.24, 16, 2048, 32, 8, 64, 8192, 131_072),
    # ---- Mistral ----------------------------------------------------------
    Architecture(
        "mistral-7b", "Mistral 7B v0.3", "mistral", 7.25, 32, 4096, 32, 8, 128, 14336, 32_768
    ),
    # Mixtral's parameter count is the total across experts; two are active per
    # token, but all of them have to be resident in VRAM.
    Architecture(
        "mixtral-8x7b", "Mixtral 8x7B", "mistral", 46.7, 32, 4096, 32, 8, 128, 14336, 32_768
    ),
    # ---- Qwen -------------------------------------------------------------
    Architecture("qwen2.5-7b", "Qwen2.5 7B", "qwen", 7.62, 28, 3584, 28, 4, 128, 18944, 131_072),
    Architecture("qwen2.5-14b", "Qwen2.5 14B", "qwen", 14.8, 48, 5120, 40, 8, 128, 13824, 131_072),
    Architecture("qwen2.5-32b", "Qwen2.5 32B", "qwen", 32.8, 64, 5120, 40, 8, 128, 27648, 131_072),
    Architecture("qwen2.5-72b", "Qwen2.5 72B", "qwen", 72.7, 80, 8192, 64, 8, 128, 29568, 131_072),
    # ---- Google Gemma -----------------------------------------------------
    # head_dim 256 against hidden 3584: deriving it would be wrong by 2x.
    Architecture("gemma-2-9b", "Gemma 2 9B", "gemma", 9.24, 42, 3584, 16, 8, 256, 14336, 8_192),
    Architecture("gemma-2-27b", "Gemma 2 27B", "gemma", 27.2, 46, 4608, 32, 16, 128, 36864, 8_192),
    # ---- Multi-head attention, kept on purpose ----------------------------
    # Without an MHA model in the catalog the GQA saving has nothing to be
    # measured against, and the tool cannot show why it matters.
    Architecture("phi-3-mini", "Phi-3 Mini 3.8B", "phi", 3.82, 32, 3072, 32, 32, 96, 8192, 131_072),
    Architecture(
        "codellama-13b", "CodeLlama 13B", "llama", 13.0, 40, 5120, 40, 40, 128, 13824, 16_384
    ),
    Architecture(
        "command-r-35b", "Command R 35B", "cohere", 35.0, 40, 8192, 64, 64, 128, 22528, 131_072
    ),
    # ---- DeepSeek ---------------------------------------------------------
    Architecture(
        "deepseek-r1-distill-llama-70b",
        "DeepSeek R1 Distill Llama 70B",
        "llama",
        70.6,
        80,
        8192,
        64,
        8,
        128,
        28672,
        131_072,
    ),
    Architecture("yi-34b", "Yi 34B", "yi", 34.4, 60, 7168, 56, 8, 128, 20480, 200_000),
)

BY_KEY: Final[dict[str, Architecture]] = {arch.key: arch for arch in ARCHITECTURES}


# Bytes per parameter by quantisation.
#
# The GGUF and AWQ figures are above their nominal bit width on purpose: a
# 4-bit quant stores scales and zero-points alongside the weights, so q4_k_m
# lands near 4.5 bits per weight in practice rather than 4. Using the nominal
# figure under-estimates a 70B model by well over a gigabyte, which is exactly
# the margin that decides whether it fits on one card.
QUANTISATION: Final[dict[str, float]] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "int4": 0.5,
    "gguf-q8_0": 1.06,
    "gguf-q6_k": 0.82,
    "gguf-q5_k_m": 0.68,
    "gguf-q4_k_m": 0.56,
    "gguf-q3_k_m": 0.43,
    "awq-4bit": 0.55,
    "gptq-4bit": 0.55,
}

# KV cache precision is set independently of the weights. Serving a model at
# INT4 with an FP16 cache is the common configuration, and quantising the
# cache to FP8 halves the term that dominates at long context.
KV_PRECISION: Final[dict[str, float]] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
}

# Runtime overhead as a multiplier on weights + cache + activations: CUDA
# context, allocator fragmentation, and the framework's own buffers.
RUNTIME_OVERHEAD: Final[dict[str, float]] = {
    "vllm": 1.10,
    "tgi": 1.15,
    "llama.cpp": 1.05,
    "transformers": 1.20,
    "sglang": 1.10,
}
