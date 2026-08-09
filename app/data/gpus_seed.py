"""GPU and accelerated-instance pricing.

Hourly on-demand USD unless `spot` is set. Hyperscaler rates are `us-east-1`
and its equivalents; neocloud rates are the published single-region price.

`vram_gb` is **per GPU**, not per instance — an 8xH100 node stores `80`, not
`640`. The instance total is `vram_gb * gpu_count`, computed where it is
needed, because per-GPU VRAM is what determines whether a model fits after
sharding and the total is what determines whether it fits at all.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

VERIFIED = date(2026, 6, 29)


class GpuSeed(NamedTuple):
    provider: str
    instance_name: str
    gpu_model: str
    gpu_count: int
    vram_gb: int
    vcpu: int | None
    ram_gb: int | None
    hourly: str
    region: str
    spot: bool
    source: str


GPUS: tuple[GpuSeed, ...] = (
    # ---- AWS ------------------------------------------------------------
    GpuSeed(
        "aws",
        "p5.48xlarge",
        "H100",
        8,
        80,
        192,
        2048,
        "98.320000",
        "us-east-1",
        False,
        "aws-ec2-pricing",
    ),
    GpuSeed(
        "aws",
        "p5.48xlarge",
        "H100",
        8,
        80,
        192,
        2048,
        "29.496000",
        "us-east-1",
        True,
        "aws-ec2-pricing",
    ),
    GpuSeed(
        "aws",
        "p4d.24xlarge",
        "A100",
        8,
        40,
        96,
        1152,
        "32.772600",
        "us-east-1",
        False,
        "aws-ec2-pricing",
    ),
    GpuSeed(
        "aws",
        "p4de.24xlarge",
        "A100",
        8,
        80,
        96,
        1152,
        "40.965700",
        "us-east-1",
        False,
        "aws-ec2-pricing",
    ),
    GpuSeed(
        "aws", "g6e.xlarge", "L40S", 1, 48, 4, 32, "1.861000", "us-east-1", False, "aws-ec2-pricing"
    ),
    GpuSeed(
        "aws",
        "g6e.12xlarge",
        "L40S",
        4,
        48,
        48,
        384,
        "10.492400",
        "us-east-1",
        False,
        "aws-ec2-pricing",
    ),
    GpuSeed(
        "aws", "g6.xlarge", "L4", 1, 24, 4, 16, "0.804500", "us-east-1", False, "aws-ec2-pricing"
    ),
    GpuSeed(
        "aws", "g5.xlarge", "A10G", 1, 24, 4, 16, "1.006000", "us-east-1", False, "aws-ec2-pricing"
    ),
    GpuSeed(
        "aws",
        "g5.12xlarge",
        "A10G",
        4,
        24,
        48,
        192,
        "5.672000",
        "us-east-1",
        False,
        "aws-ec2-pricing",
    ),
    GpuSeed(
        "aws", "g4dn.xlarge", "T4", 1, 16, 4, 16, "0.526000", "us-east-1", False, "aws-ec2-pricing"
    ),
    # ---- Google Cloud ---------------------------------------------------
    GpuSeed(
        "gcp",
        "a3-highgpu-8g",
        "H100",
        8,
        80,
        208,
        1872,
        "88.246000",
        "us-central1",
        False,
        "gcp-compute-pricing",
    ),
    GpuSeed(
        "gcp",
        "a3-megagpu-8g",
        "H100",
        8,
        80,
        208,
        1872,
        "94.480000",
        "us-central1",
        False,
        "gcp-compute-pricing",
    ),
    GpuSeed(
        "gcp",
        "a2-highgpu-1g",
        "A100",
        1,
        40,
        12,
        85,
        "3.673900",
        "us-central1",
        False,
        "gcp-compute-pricing",
    ),
    GpuSeed(
        "gcp",
        "a2-ultragpu-1g",
        "A100",
        1,
        80,
        12,
        170,
        "5.068700",
        "us-central1",
        False,
        "gcp-compute-pricing",
    ),
    GpuSeed(
        "gcp",
        "g2-standard-4",
        "L4",
        1,
        24,
        4,
        16,
        "0.855000",
        "us-central1",
        False,
        "gcp-compute-pricing",
    ),
    GpuSeed(
        "gcp",
        "g2-standard-48",
        "L4",
        4,
        24,
        48,
        192,
        "4.964000",
        "us-central1",
        False,
        "gcp-compute-pricing",
    ),
    # ---- Azure ----------------------------------------------------------
    GpuSeed(
        "azure",
        "ND96isr_H100_v5",
        "H100",
        8,
        80,
        96,
        1900,
        "98.320000",
        "eastus",
        False,
        "azure-vm-pricing",
    ),
    GpuSeed(
        "azure",
        "ND96amsr_A100_v4",
        "A100",
        8,
        80,
        96,
        1900,
        "32.770000",
        "eastus",
        False,
        "azure-vm-pricing",
    ),
    GpuSeed(
        "azure",
        "NC24ads_A100_v4",
        "A100",
        1,
        80,
        24,
        220,
        "3.673000",
        "eastus",
        False,
        "azure-vm-pricing",
    ),
    GpuSeed(
        "azure", "NC4as_T4_v3", "T4", 1, 16, 4, 28, "0.526000", "eastus", False, "azure-vm-pricing"
    ),
    # ---- Lambda Labs ----------------------------------------------------
    GpuSeed(
        "lambda",
        "gpu_8x_h100_sxm5",
        "H100",
        8,
        80,
        208,
        1800,
        "23.920000",
        "us-west-1",
        False,
        "lambda-labs-pricing",
    ),
    GpuSeed(
        "lambda",
        "gpu_1x_h100_pcie",
        "H100",
        1,
        80,
        26,
        200,
        "2.490000",
        "us-west-1",
        False,
        "lambda-labs-pricing",
    ),
    GpuSeed(
        "lambda",
        "gpu_1x_a100_sxm4",
        "A100",
        1,
        80,
        30,
        220,
        "1.790000",
        "us-west-1",
        False,
        "lambda-labs-pricing",
    ),
    GpuSeed(
        "lambda",
        "gpu_1x_a10",
        "A10",
        1,
        24,
        30,
        200,
        "0.750000",
        "us-west-1",
        False,
        "lambda-labs-pricing",
    ),
    # ---- RunPod ---------------------------------------------------------
    GpuSeed(
        "runpod", "H200 SXM", "H200", 1, 141, 24, 251, "3.590000", "global", False, "runpod-pricing"
    ),
    GpuSeed(
        "runpod", "H100 SXM", "H100", 1, 80, 20, 251, "2.690000", "global", False, "runpod-pricing"
    ),
    GpuSeed(
        "runpod", "H100 SXM", "H100", 1, 80, 20, 251, "1.650000", "global", True, "runpod-pricing"
    ),
    GpuSeed(
        "runpod", "A100 SXM", "A100", 1, 80, 16, 125, "1.890000", "global", False, "runpod-pricing"
    ),
    GpuSeed(
        "runpod",
        "RTX 4090",
        "RTX 4090",
        1,
        24,
        8,
        32,
        "0.690000",
        "global",
        False,
        "runpod-pricing",
    ),
    GpuSeed(
        "runpod",
        "RTX A6000",
        "RTX A6000",
        1,
        48,
        8,
        50,
        "0.760000",
        "global",
        False,
        "runpod-pricing",
    ),
    GpuSeed("runpod", "L40S", "L40S", 1, 48, 16, 62, "0.860000", "global", False, "runpod-pricing"),
    # ---- Vast.ai (marketplace medians) ----------------------------------
    GpuSeed(
        "vast",
        "H100 SXM (median)",
        "H100",
        1,
        80,
        16,
        128,
        "1.930000",
        "global",
        False,
        "vast-ai-pricing",
    ),
    GpuSeed(
        "vast",
        "A100 SXM (median)",
        "A100",
        1,
        80,
        12,
        96,
        "1.100000",
        "global",
        False,
        "vast-ai-pricing",
    ),
    GpuSeed(
        "vast",
        "RTX 4090 (median)",
        "RTX 4090",
        1,
        24,
        8,
        32,
        "0.350000",
        "global",
        False,
        "vast-ai-pricing",
    ),
    GpuSeed(
        "vast",
        "RTX 3090 (median)",
        "RTX 3090",
        1,
        24,
        8,
        32,
        "0.220000",
        "global",
        False,
        "vast-ai-pricing",
    ),
)
