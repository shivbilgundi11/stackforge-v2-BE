"""Real token counts, and an honest label when they are not.

Replaces `ceil(chars / 4)`. The `tokenizer` column on `model_pricing` drives
the selection, so adding a model is a data change:

    tiktoken:<encoding>   OpenAI models, counted with the real BPE
    anthropic:api         counted by the provider's own count_tokens endpoint
    hf:<repo>             open-weight models, counted with the HF tokenizer
    <anything else>       no tokenizer available — heuristic, and it says so

**`tiktoken` is never used for a Claude model.** It is OpenAI's tokenizer and
undercounts Claude by 15-20% on prose and far more on code. Anthropic models
go to `count_tokens`, which is the only accurate answer, and which is why this
module needs a client at all.

`method` is returned to the caller and goes on the API response, not into a
log line. The person reaching for a token calculator is exactly the person who
needs to know whether the number is measured or estimated — a calculator that
silently returns a bad number is worse than one that says which it returned.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Final, NamedTuple

from app.core.logging import get_logger
from app.schemas.catalog import ModelOut
from app.services import ai_service

logger = get_logger("tokenizer")

#: ~4 characters per token for English prose, with a word-based floor that
#: catches code and URLs the character rule under-counts. Only ever used when
#: no real tokenizer is reachable, and always reported as `heuristic`.
CHARS_PER_TOKEN: Final = 4
TOKENS_PER_WORD: Final = 1.3

MEASURED: Final = "tokenizer"
ESTIMATED: Final = "heuristic"

#: Loaded HF tokenizers, by repo. Loading one is slow and hits the network the
#: first time; a calculator that pays that on every keystroke is unusable.
_hf_cache: dict[str, Any] = {}
#: Repos that failed to load. Retrying a missing repo on every request turns
#: one bad seed row into a per-request network timeout.
_hf_failed: set[str] = set()
_tiktoken_cache: dict[str, Any] = {}


class TokenCount(NamedTuple):
    tokens: int
    #: `tokenizer` or `heuristic`. Goes on the response.
    method: str
    #: What actually produced the number, for the warning text.
    detail: str


def heuristic(text: str) -> TokenCount:
    if not text:
        return TokenCount(0, ESTIMATED, "empty input")
    estimate = max(len(text) / CHARS_PER_TOKEN, len(text.split()) * TOKENS_PER_WORD)
    return TokenCount(math.ceil(estimate), ESTIMATED, "character and word heuristic")


async def count(text: str, *, model: ModelOut | None = None) -> TokenCount:
    """Count `text` for `model`, falling back to the heuristic and saying so."""
    spec = (model.tokenizer if model else None) or ""

    if spec.startswith("tiktoken:"):
        return await asyncio.to_thread(_tiktoken_count, text, spec.split(":", 1)[1])
    if spec.startswith("anthropic"):
        return await _anthropic_count(text, model)
    if spec.startswith("hf:"):
        return await asyncio.to_thread(_hf_count, text, spec.split(":", 1)[1])

    return heuristic(text)


def _tiktoken_count(text: str, encoding_name: str) -> TokenCount:
    """OpenAI's BPE, for OpenAI models only."""
    try:
        encoding = _tiktoken_cache.get(encoding_name)
        if encoding is None:
            import tiktoken

            encoding = tiktoken.get_encoding(encoding_name)
            _tiktoken_cache[encoding_name] = encoding
        return TokenCount(len(encoding.encode(text)), MEASURED, f"tiktoken {encoding_name}")
    except Exception as exc:
        logger.warning("tokenizer.tiktoken_failed", encoding=encoding_name, error=str(exc))
        return heuristic(text)


async def _anthropic_count(text: str, model: ModelOut | None) -> TokenCount:
    """The provider's own count. Nothing local is accurate for these models.

    Without a key there is no way to count a Claude model correctly, and
    guessing with someone else's tokenizer would be worse than saying so —
    the heuristic at least reports itself.
    """
    client = ai_service.get_client()
    if client is None:
        return heuristic(text)

    model_id = model.model_id if model else "claude-opus-5"
    try:
        response = await client.messages.count_tokens(
            model=model_id,
            messages=[{"role": "user", "content": text}],
        )
        return TokenCount(int(response.input_tokens), MEASURED, f"{model_id} count_tokens")
    except Exception as exc:
        logger.warning("tokenizer.anthropic_failed", model=model_id, error=str(exc))
        return heuristic(text)


def _hf_count(text: str, repo: str) -> TokenCount:
    """The open-weight model's own tokenizer, downloaded once and cached."""
    if repo in _hf_failed:
        return heuristic(text)
    try:
        tokenizer = _hf_cache.get(repo)
        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_pretrained(repo)
            _hf_cache[repo] = tokenizer
        return TokenCount(len(tokenizer.encode(text).ids), MEASURED, f"{repo} tokenizer")
    except Exception as exc:
        # Gated repos and offline environments both land here. Remembered, so
        # the next request does not repeat the network wait.
        _hf_failed.add(repo)
        logger.warning("tokenizer.hf_failed", repo=repo, error=str(exc))
        return heuristic(text)


def reset_caches() -> None:
    """Test seam."""
    _hf_cache.clear()
    _hf_failed.clear()
    _tiktoken_cache.clear()
