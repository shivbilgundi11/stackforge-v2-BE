"""Token counting, and the label that says how it was counted.

`method` is the point of this module. A tool that returns a wrong number
confidently is worse than one that returns the same number and says it is an
estimate, so every path here asserts the label as well as the count.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from app.core.config import settings
from app.schemas.catalog import ModelOut, ProvenanceOut
from app.services import tokenizer_service

SAMPLE = "The quick brown fox jumps over the lazy dog. " * 20


@pytest.fixture(autouse=True)
def _clean():
    tokenizer_service.reset_caches()
    yield
    tokenizer_service.reset_caches()


def _model(model_id: str, tokenizer: str | None) -> ModelOut:
    return ModelOut(
        id=f"mdl_{model_id}",
        provider="test",
        model_id=model_id,
        display_name=model_id,
        family="chat",
        input_cost_per_1k="0.001",
        status="active",
        tokenizer=tokenizer,
        provenance=ProvenanceOut(
            last_verified_at=datetime(2026, 8, 1, tzinfo=UTC),
            age_days=9,
            variant="fresh",
            source_name="Test",
            source_url="https://example.com",
            source_kind="manual",
        ),
    )


async def test_an_openai_model_is_counted_with_the_real_bpe() -> None:
    result = await tokenizer_service.count(SAMPLE, model=_model("gpt-4o", "tiktoken:o200k_base"))

    assert result.method == tokenizer_service.MEASURED
    assert "tiktoken" in result.detail

    import tiktoken

    assert result.tokens == len(tiktoken.get_encoding("o200k_base").encode(SAMPLE))


async def test_the_openai_count_differs_from_the_heuristic() -> None:
    """If they agreed, the tokenizer would not be earning its dependency."""
    measured = await tokenizer_service.count(SAMPLE, model=_model("gpt-4o", "tiktoken:o200k_base"))
    estimated = tokenizer_service.heuristic(SAMPLE)

    assert measured.tokens != estimated.tokens


async def test_a_claude_model_is_counted_by_the_provider_not_by_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tiktoken is OpenAI's tokenizer and undercounts Claude badly. The only
    accurate count for these models is the provider's own endpoint."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    seen: dict[str, Any] = {}

    class _Messages:
        async def count_tokens(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return SimpleNamespace(input_tokens=4242)

    tokenizer_service.set_anthropic_client(SimpleNamespace(messages=_Messages()))

    result = await tokenizer_service.count(SAMPLE, model=_model("claude-opus-5", "anthropic:api"))

    assert result.tokens == 4242
    assert result.method == tokenizer_service.MEASURED
    assert seen["model"] == "claude-opus-5"


async def test_a_claude_model_without_a_key_falls_back_and_says_so() -> None:
    """Counting is the only thing the Anthropic key still buys - synthesis
    runs on Gemini - so its absence has to cost a labelled heuristic and
    nothing else."""
    result = await tokenizer_service.count(SAMPLE, model=_model("claude-opus-5", "anthropic:api"))

    assert result.method == tokenizer_service.ESTIMATED


async def test_counting_does_not_ride_on_the_generation_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two keys buy different things from different vendors. A deploy with
    a Gemini key and no Anthropic key must get full synthesis and an honest
    heuristic here, not a crash from counting Claude tokens with neither."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    result = await tokenizer_service.count(SAMPLE, model=_model("claude-opus-5", "anthropic:api"))

    assert result.method == tokenizer_service.ESTIMATED
    assert result.tokens > 0


async def test_a_provider_error_degrades_rather_than_failing_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")

    class _Messages:
        async def count_tokens(self, **kwargs: Any) -> Any:
            raise RuntimeError("upstream is down")

    tokenizer_service.set_anthropic_client(SimpleNamespace(messages=_Messages()))

    result = await tokenizer_service.count(SAMPLE, model=_model("claude-opus-5", "anthropic:api"))
    assert result.method == tokenizer_service.ESTIMATED
    assert result.tokens > 0


async def test_a_model_with_no_tokenizer_reports_the_heuristic() -> None:
    result = await tokenizer_service.count(SAMPLE, model=_model("gemini-3-flash", "google:api"))

    assert result.method == tokenizer_service.ESTIMATED


async def test_no_model_at_all_reports_the_heuristic() -> None:
    assert (await tokenizer_service.count(SAMPLE)).method == tokenizer_service.ESTIMATED


async def test_an_unreachable_open_model_is_only_attempted_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing or gated repo must not become a network wait on every
    keystroke."""
    attempts = {"count": 0}

    def _explode(_repo: str) -> Any:
        attempts["count"] += 1
        raise OSError("no network")

    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_pretrained", staticmethod(_explode), raising=False
    )
    model = _model("llama-4", "hf:meta-llama/Does-Not-Exist")

    first = await tokenizer_service.count(SAMPLE, model=model)
    second = await tokenizer_service.count(SAMPLE, model=model)

    assert first.method == second.method == tokenizer_service.ESTIMATED
    assert attempts["count"] == 1


async def test_an_open_model_uses_its_own_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Encoding:
        ids: ClassVar[list[int]] = list(range(137))

    class _Tokenizer:
        def encode(self, _text: str) -> Any:
            return _Encoding()

    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_pretrained",
        staticmethod(lambda _repo: _Tokenizer()),
        raising=False,
    )

    result = await tokenizer_service.count(
        SAMPLE, model=_model("llama-4", "hf:meta-llama/Llama-4-Scout")
    )

    assert result.tokens == 137
    assert result.method == tokenizer_service.MEASURED


def test_the_heuristic_never_undercounts_code_by_the_character_rule_alone() -> None:
    """The word floor exists for code and URLs, where 4-chars-per-token is
    optimistic."""
    dense = " ".join(["x"] * 200)
    assert tokenizer_service.heuristic(dense).tokens > len(dense) // 4


def test_empty_input_is_zero_not_an_error() -> None:
    assert tokenizer_service.heuristic("").tokens == 0
