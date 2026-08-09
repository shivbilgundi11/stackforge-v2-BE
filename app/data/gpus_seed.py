"""GPU and accelerated-instance pricing.

Hourly on-demand USD unless `spot` is set. Hyperscaler rates are `us-east-1`
and its equivalents; neocloud rates are the published single-region price.

`vram_gb` is **per GPU**, not per instance - an 8xH100 node stores `80`, not
`640`. The instance total is `vram_gb * gpu_count`, computed where it is
needed, because per-GPU VRAM is what determines whether a model fits after
sharding and the total is what determines whether it fits at all.

Two verification dates, and the split is deliberate (D-16).

`VERIFIED_LIVE` rows were read from a price the vendor *publishes as a list
price* and can be checked again the same way: the AWS and Azure pricing APIs,
and the GCP, Lambda, and RunPod rate cards.

`VERIFIED_CARRIED` rows are auction-cleared - EC2 spot and the Vast.ai
marketplace - and clear on supply and demand, minute to minute. A hand-checked
snapshot of one of those is not a list price that has gone slightly stale; it
is a different kind of number, and refreshing its date would claim a currency
it cannot have. They keep their original date on purpose, so the freshness chip
renders them stale, because they are.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

VERIFIED_LIVE = date(2026, 8, 9)
VERIFIED_CARRIED = date(2026, 6, 29)


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
    verified: date = VERIFIED_LIVE


GPUS: tuple[GpuSeed, ...] = (
    # ---- AWS --------------------------------------------------------------
    # On-demand read from the pricing feed the EC2 pricing page itself calls,
    # us-east-1/Linux. The P-family carried pre-price-cut rates: p5 was listed
    # at 98.32 against an actual 55.04, which made every self-host-vs-API
    # break-even in the infra planner wrong by nearly a factor of two.
    GpuSeed(
        "aws",
        "p5.48xlarge",
        "H100",
        8,
        80,
        192,
        2048,
        "55.040000",
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
        # Auction-cleared, and left at its original date. The public spot feed
        # still quotes 57.76 here, which is above the corrected on-demand rate
        # and therefore impossible - AWS caps spot at on-demand - so the feed
        # is stale against the price cut and is not a source worth copying.
        "29.496000",
        "us-east-1",
        True,
        "aws-ec2-pricing",
        VERIFIED_CARRIED,
    ),
    GpuSeed(
        "aws",
        "p4d.24xlarge",
        "A100",
        8,
        40,
        96,
        1152,
        "21.957642",
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
        "27.447050",
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
        "10.492640",
        "us-east-1",
        False,
        "aws-ec2-pricing",
    ),
    GpuSeed(
        "aws", "g6.xlarge", "L4", 1, 24, 4, 16, "0.804800", "us-east-1", False, "aws-ec2-pricing"
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
    # ---- Google Cloud -----------------------------------------------------
    GpuSeed(
        "gcp",
        "a3-highgpu-8g",
        "H100",
        8,
        80,
        208,
        1872,
        "88.490000",
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
        "93.400713",
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
        "3.673385",
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
        "5.068798",
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
        "0.706832",
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
        "4.001665",
        "us-central1",
        False,
        "gcp-compute-pricing",
    ),
    # ---- Azure ------------------------------------------------------------
    # Read from the public retail-prices API, eastus/Linux/Consumption. All
    # four were already correct.
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
    # ---- Lambda Labs ------------------------------------------------------
    # The rate card quotes per-GPU-hour; multi-GPU rows below are the node
    # total. Every Lambda row had drifted low, the 8xH100 node by a third.
    GpuSeed(
        "lambda",
        "gpu_8x_h100_sxm5",
        "H100",
        8,
        80,
        208,
        1800,
        # 8 x $3.99/GPU/hr.
        "31.920000",
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
        "3.290000",
        "us-west-1",
        False,
        "lambda-labs-pricing",
    ),
    GpuSeed(
        "lambda",
        "gpu_1x_a100_sxm4",
        "A100",
        1,
        # 40, not 80: Lambda offers no single-GPU 80GB A100. The 80GB part is
        # sold only as the 8x node, and claiming 80GB here would have let the
        # VRAM estimator fit a model onto an instance nobody can rent.
        40,
        30,
        220,
        "1.990000",
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
        "1.290000",
        "us-west-1",
        False,
        "lambda-labs-pricing",
    ),
    # ---- RunPod -----------------------------------------------------------
    # RunPod publishes two tiers, and the seed previously mixed them: some
    # rows were Community, some Secure, with nothing recording which. Mapped
    # deliberately now - Secure Cloud is the dedicated tier and reads as
    # on-demand, Community is the cheap interruptible one and reads as spot.
    GpuSeed(
        "runpod", "H200 SXM", "H200", 1, 141, 24, 251, "4.390000", "global", False, "runpod-pricing"
    ),
    GpuSeed(
        "runpod", "H100 SXM", "H100", 1, 80, 20, 251, "2.990000", "global", False, "runpod-pricing"
    ),
    GpuSeed(
        "runpod", "H100 SXM", "H100", 1, 80, 20, 251, "2.690000", "global", True, "runpod-pricing"
    ),
    GpuSeed(
        "runpod", "A100 SXM", "A100", 1, 80, 16, 125, "1.490000", "global", False, "runpod-pricing"
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
        "0.490000",
        "global",
        False,
        "runpod-pricing",
    ),
    GpuSeed("runpod", "L40S", "L40S", 1, 48, 16, 62, "0.990000", "global", False, "runpod-pricing"),
    # ---- Vast.ai (marketplace medians) ------------------------------------
    # Left at their original date, deliberately. Vast is an auction across
    # 40+ datacentres with no published rate card, so there is nothing to
    # verify against - only a live figure that is different by the time it is
    # read. Dating these to today would dress a guess as a check.
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
        VERIFIED_CARRIED,
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
        VERIFIED_CARRIED,
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
        VERIFIED_CARRIED,
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
        VERIFIED_CARRIED,
    ),
)
