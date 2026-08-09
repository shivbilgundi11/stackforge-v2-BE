"""Where every priced row came from.

Each entry is a page a human can open to check a number. `kind` records how the
verification job reads it: `api` for a machine-readable endpoint, `scrape` for
an HTML pricing page, `manual` for anything that needs a person.
"""

from __future__ import annotations

from typing import NamedTuple


class SourceSeed(NamedTuple):
    slug: str
    name: str
    url: str
    kind: str


SOURCES: tuple[SourceSeed, ...] = (
    SourceSeed(
        "anthropic-pricing",
        "Anthropic — API pricing",
        "https://platform.claude.com/docs/en/pricing",
        "scrape",
    ),
    SourceSeed(
        "openai-pricing",
        "OpenAI — API pricing",
        "https://developers.openai.com/api/docs/pricing",
        "scrape",
    ),
    SourceSeed(
        "google-gemini-pricing",
        "Google — Gemini API pricing",
        "https://ai.google.dev/gemini-api/docs/pricing",
        "scrape",
    ),
    SourceSeed(
        "mistral-pricing",
        "Mistral AI — pricing",
        "https://mistral.ai/pricing",
        "scrape",
    ),
    SourceSeed(
        "cohere-pricing",
        "Cohere — pricing",
        "https://cohere.com/pricing",
        "scrape",
    ),
    SourceSeed(
        "deepseek-pricing",
        "DeepSeek — API pricing",
        "https://api-docs.deepseek.com/quick_start/pricing",
        "scrape",
    ),
    SourceSeed(
        "xai-pricing",
        "xAI — models and pricing",
        "https://docs.x.ai/docs/models",
        "scrape",
    ),
    SourceSeed(
        "together-pricing",
        "Together AI — pricing",
        "https://www.together.ai/pricing",
        "scrape",
    ),
    SourceSeed(
        "voyage-pricing",
        "Voyage AI — pricing",
        "https://docs.voyageai.com/docs/pricing",
        "scrape",
    ),
    SourceSeed(
        "jina-pricing",
        "Jina AI — pricing",
        "https://jina.ai/embeddings",
        "scrape",
    ),
    SourceSeed(
        "aws-bedrock-pricing",
        "AWS — Bedrock pricing",
        "https://aws.amazon.com/bedrock/pricing/",
        "scrape",
    ),
    SourceSeed(
        "aws-ec2-pricing",
        "AWS — EC2 accelerated computing pricing",
        "https://aws.amazon.com/ec2/pricing/on-demand/",
        "api",
    ),
    SourceSeed(
        "gcp-compute-pricing",
        "Google Cloud — GPU pricing",
        "https://cloud.google.com/compute/gpus-pricing",
        "scrape",
    ),
    SourceSeed(
        "azure-vm-pricing",
        "Azure — GPU virtual machine pricing",
        "https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/",
        "scrape",
    ),
    SourceSeed(
        "lambda-labs-pricing",
        "Lambda Labs — GPU cloud pricing",
        "https://lambdalabs.com/service/gpu-cloud",
        "scrape",
    ),
    SourceSeed(
        "runpod-pricing",
        "RunPod — GPU pricing",
        "https://www.runpod.io/pricing",
        "scrape",
    ),
    SourceSeed(
        "vast-ai-pricing",
        "Vast.ai — GPU marketplace pricing",
        "https://vast.ai/pricing",
        "scrape",
    ),
    SourceSeed(
        "editorial-review",
        "StackForge editorial review",
        "https://stackforge.dev/methodology",
        "manual",
    ),
)

SOURCES_BY_SLUG = {source.slug: source for source in SOURCES}
