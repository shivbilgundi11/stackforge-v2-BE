"""Every tool that claims a model wrote something, calling one (M16).

M16 exists because the old build's marketing promised synthesis and the audit
found no model call anywhere. The prompt registry has carried eight prompts
since; five of them were never reached from an endpoint, which is the same
gap one layer further in — a prompt nothing calls is a claim nothing keeps.

So these tests assert the wiring rather than the prose: for each tool that is
meant to synthesise, that a request produces a model call, that what came back
reaches the response the page renders, and that the run says `hybrid` so the
provenance chip names the model. The text itself is not asserted — the model
picks that, and a test that pinned it would fail on a prompt improvement.

The provider is stubbed throughout. A test that made live calls would be
non-deterministic, billable, and — on a free tier metered in requests per day
— self-limiting: twenty of them and the suite stops working until tomorrow.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.ai import AiCall, AiOutcome
from app.models.billing import Metric
from app.models.user import Plan
from app.services import ai_prompts, ai_service
from tests.conftest import set_limit

pytestmark = pytest.mark.usefixtures("seeded_catalog")


class _FakeGemini:
    """Answers every call with the reply keyed to the purpose asked for.

    Keyed rather than scripted in order, because the Architect makes two calls
    in one request and a positional stub would silently hand the roadmap
    prompt the assessment's answer — which the schema would accept and the
    page would render as an empty roadmap.

    The purpose is recovered from the schema rather than from the URL: every
    prompt in the registry has its own, and matching on it is what makes a
    stub wired to the wrong prompt fail loudly here instead of quietly in the
    assertion twenty lines down.
    """

    def __init__(self, replies: dict[str, dict[str, Any]]) -> None:
        self._by_schema = {
            json.dumps(ai_prompts.REGISTRY[purpose].schema, sort_keys=True): reply
            for purpose, reply in replies.items()
        }
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> _FakeGemini:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        schema = kwargs["json"]["generationConfig"]["responseJsonSchema"]
        reply = self._by_schema.get(json.dumps(schema, sort_keys=True))
        if reply is None:
            raise AssertionError(f"no stubbed reply for the schema sent to {url}")
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(reply)}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 900,
                    "candidatesTokenCount": 300,
                    "thoughtsTokenCount": 400,
                },
            },
        )


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., _FakeGemini]]:
    """A stubbed Gemini, and a key that makes the service believe in it."""

    def install(replies: dict[str, dict[str, Any]]) -> _FakeGemini:
        monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
        client = _FakeGemini(replies)
        monkeypatch.setattr(ai_service.httpx, "AsyncClient", lambda **_: client)
        return client

    return install


async def _purposes(db: AsyncSession) -> list[tuple[str, AiOutcome]]:
    rows = await db.execute(select(AiCall.purpose, AiCall.outcome).order_by(AiCall.created_at))
    return [(purpose, outcome) for purpose, outcome in rows.all()]


SYNTHESIS_REPLY = {
    "recommended_rank": "1",
    "confidence": "high",
    "summary": "A managed-model RAG stack sized for one team.",
    "why": "It clears every hard constraint and costs the least of the ones that do.",
    "trade_offs": ["You are renting the model rather than owning it."],
    "switch_when": ["Move to the runner-up once the corpus outgrows one node."],
    "risks": [{"risk": "Vendor pricing moves", "severity": "medium", "mitigation": "Abstract it"}],
}

ROADMAP_REPLY = {
    "steps": [
        {
            "title": f"Step {index}",
            "detail": f"What happens in step {index}.",
            "effort": "2-3 days",
            "depends_on": "the previous step",
            "done_when": "it is measurably working",
        }
        for index in range(1, 6)
    ]
}


# ── stack architect ──────────────────────────────────────────────────────────


async def test_the_architect_roadmap_is_written_by_a_model_and_reaches_the_page(
    client: AsyncClient, db: AsyncSession, gemini: Any
) -> None:
    """The panel this fills was empty for the product's whole life: the table
    shipped as `[]` and the exported document said "roadmap unavailable"."""
    stub = gemini({"stack_synthesis": SYNTHESIS_REPLY, "roadmap": ROADMAP_REPLY})

    response = await client.post("/api/v1/architect/recommend", json={})
    assert response.status_code == 200, response.text
    data = response.json()["data"]

    # Two passes, one run. The usage line is the sum of both, and `hybrid`
    # means at least one model actually contributed.
    assert len(stub.calls) == 2
    assert data["source"] == "hybrid"
    assert data["ai"]["model"]
    assert data["metrics"]["summary"] == SYNTHESIS_REPLY["summary"]

    steps = data["tables"]["roadmap"]
    assert len(steps) == 5
    assert steps[0]["title"] == "Step 1"
    assert steps[0]["done_when"]

    assert await _purposes(db) == [
        ("stack_synthesis", AiOutcome.SUCCESS),
        ("roadmap", AiOutcome.SUCCESS),
    ]


DIMENSION_KEYS = (
    "cost_efficiency",
    "scalability",
    "developer_experience",
    "production_readiness",
    "security_readiness",
    "vendor_lock_in",
    "integration_compatibility",
    "deployment_complexity",
    "community_maturity",
    "documentation_quality",
)


async def test_the_headline_score_stays_the_engines_through_synthesis(
    client: AsyncClient, gemini: Any
) -> None:
    """The number on the ring and the number in the prose are one number.

    The schema used to ask the model for its own ten dimension scores, and the
    route recomputed the headline from them — *after* handing the model the
    engine's total as grounding and telling it never to restate a number
    differently. The page then showed one total above a paragraph quoting
    another, and the breakdown rows summed to neither. A model that answers in
    the old shape must not move the arithmetic.
    """
    gemini(
        {
            "stack_synthesis": {
                **SYNTHESIS_REPLY,
                "score_breakdown": [{"key": key, "score": 10} for key in DIMENSION_KEYS],
            },
            "roadmap": ROADMAP_REPLY,
        }
    )

    response = await client.post("/api/v1/architect/recommend", json={})
    data = response.json()["data"]
    assert data["source"] == "hybrid"

    score = float(data["metrics"]["score"])
    assert score < 100, "a perfect ten on every dimension reached the headline"
    total = sum(float(row["contribution"]) for row in data["tables"]["score_breakdown"])
    assert abs(total - score) < 0.6, "the breakdown no longer sums to the headline"
    assert all(float(row["score"]) < 10 for row in data["tables"]["score_breakdown"])


async def test_the_model_can_pick_a_runner_up_and_the_whole_page_follows(
    client: AsyncClient, gemini: Any
) -> None:
    """M15 layer 2: the engine ranks, the model selects among what it ranked.

    The schema always asked for that selection and nothing ever read it, so a
    rationale arguing for the runner-up shipped above the leader's component
    table. Selecting has to move every panel at once — score, components,
    diagram, alternatives and the export — or the page contradicts itself
    somewhere new instead.
    """
    gemini(
        {
            "stack_synthesis": {**SYNTHESIS_REPLY, "recommended_rank": "2"},
            "roadmap": ROADMAP_REPLY,
        }
    )

    response = await client.post("/api/v1/architect/recommend", json={})
    data = response.json()["data"]
    assert data["source"] == "hybrid"

    # The stack the engine led with is now the alternative, and the one the
    # model chose is no longer offered as one.
    alternatives = data["tables"]["alternatives"]
    ranks = [row["rank"] for row in alternatives]
    assert 1 in ranks, "the model's pick did not reach the page"
    assert 2 not in ranks, "the recommendation is still listed against itself"

    # Its own score, not the leader's — and the leader still outranks it,
    # which is the trade the rationale is there to justify.
    leader = next(row for row in alternatives if row["rank"] == 1)
    assert float(leader["score"]) >= float(data["metrics"]["score"])

    diagram = next(a for a in data["artifacts"] if a["format"] == "mermaid")["content"]
    document = next(a for a in data["artifacts"] if a["type"] == "architecture")["content"]
    for row in data["tables"]["components"]:
        assert row["name"] in diagram, "the diagram is of the stack that lost"
        assert row["name"] in document, "the export is of the stack that lost"


async def test_a_rank_the_engine_never_offered_leaves_the_leader_alone(
    client: AsyncClient, gemini: Any
) -> None:
    """D-06: the fallback for a malformed answer is the deterministic one."""
    gemini(
        {
            "stack_synthesis": {**SYNTHESIS_REPLY, "recommended_rank": "3000"},
            "roadmap": ROADMAP_REPLY,
        }
    )

    response = await client.post("/api/v1/architect/recommend", json={})
    data = response.json()["data"]

    assert data["source"] == "hybrid"
    assert 1 not in [row["rank"] for row in data["tables"]["alternatives"]]
    assert data["tables"]["components"]


async def test_one_failed_pass_still_leaves_the_other_on_the_page(
    client: AsyncClient, db: AsyncSession, gemini: Any
) -> None:
    """Partial enrichment is the normal outcome when an allowance runs out
    mid-run — the free tier is metered in requests per day, so the second call
    of a two-call request is exactly where it runs out. One written section is
    worth more than none."""
    gemini({"roadmap": ROADMAP_REPLY})  # nothing stubbed for the assessment

    response = await client.post("/api/v1/architect/recommend", json={})
    data = response.json()["data"]

    assert data["source"] == "hybrid"
    assert len(data["tables"]["roadmap"]) == 5
    # The rule engine wrote the summary, because the pass that would have
    # replaced it failed.
    assert data["metrics"]["summary"]
    assert await _purposes(db) == [
        ("stack_synthesis", AiOutcome.API_ERROR),
        ("roadmap", AiOutcome.SUCCESS),
    ]


async def test_the_exported_document_carries_the_same_roadmap_as_the_page(
    client: AsyncClient, gemini: Any
) -> None:
    """The document is built during `compute`, before any model has answered.
    Filling only the table would leave a download that disagrees with the page
    it was downloaded from — the one failure this artifact exists to avoid."""
    gemini({"stack_synthesis": SYNTHESIS_REPLY, "roadmap": ROADMAP_REPLY})

    response = await client.post("/api/v1/architect/recommend", json={})
    data = response.json()["data"]

    document = next(a for a in data["artifacts"] if a["type"] == "architecture")["content"]
    assert "Roadmap unavailable" not in document
    for step in data["tables"]["roadmap"]:
        assert step["title"] in document


async def test_a_failed_roadmap_leaves_the_recommendation_whole(
    client: AsyncClient, gemini: Any
) -> None:
    """D-06: the rule engine is the product. Every figure on the page is
    computed before a model is asked anything, so a synthesis failure costs
    the commentary and nothing else."""
    stub = gemini({})  # no reply for any purpose — every call raises

    response = await client.post("/api/v1/architect/recommend", json={})
    assert response.status_code == 200
    data = response.json()["data"]

    assert stub.calls  # it was attempted
    assert data["source"] == "rule_based"
    assert data["ai"] is None
    assert data["tables"]["roadmap"] == []
    assert data["tables"]["components"]
    assert float(data["metrics"]["score"]) > 0


async def test_the_compatibility_checker_explains_what_the_weakest_pair_costs(
    client: AsyncClient, gemini: Any
) -> None:
    gemini(
        {
            "compatibility_rationale": {
                "summary": "These three sit together without much glue.",
                "weakest_pair_impact": "You will write your own sync job.",
            }
        }
    )

    response = await client.post(
        "/api/v1/architect/score",
        json={
            "component_slugs": ["pgvector", "langgraph", "redis"],
            "monthly_budget": 2000,
            "scale_target": "medium",
            "sensitivity": "internal",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]

    assert data["source"] == "hybrid"
    assert data["metrics"]["weakest_pair_impact"] == "You will write your own sync job."
    # The scores stay the engine's.
    assert len(data["tables"]["score_breakdown"]) == 10


# ── compare centre ───────────────────────────────────────────────────────────


async def test_a_comparison_rewrites_the_verdict_and_keeps_the_arithmetic(
    client: AsyncClient, gemini: Any
) -> None:
    gemini(
        {
            "comparison_rationale": {
                "why": "It wins on the two criteria this profile weights most.",
                "switch_when": "Pick the runner-up once latency matters more than price.",
            }
        }
    )

    response = await client.post(
        "/api/v1/tools/compare/stacks",
        json={"archetypes": ["serverless", "self-hosted"], "priority": "balanced"},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]

    rationale = data["tables"]["rationale"]
    whys = [row for row in rationale if row["kind"] == "why"]
    assert len(whys) == 1, "two answers to the same question reads as indecision"
    assert whys[0]["text"] == "It wins on the two criteria this profile weights most."

    switch_when = [row["text"] for row in rationale if row["kind"] == "switch_when"]
    assert "Pick the runner-up once latency matters more than price." in switch_when
    assert len(switch_when) > 1, "the engine's threshold and the model's situation are both advice"

    assert data["source"] == "hybrid"
    assert data["tables"]["matrix"]


async def test_every_comparison_is_wired_to_the_same_prompt(
    client: AsyncClient, db: AsyncSession, gemini: Any
) -> None:
    """Four endpoints, one prompt. Wiring three of four is the failure mode
    that survives review, because the page looks identical either way."""
    gemini({"comparison_rationale": {"why": "because", "switch_when": "when"}})
    # An anonymous visitor gets two AI calls a day, and this test needs four
    # in one request cycle. Raising the quota is the operator action, not a
    # monkeypatch — the same write M20 exposes.
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=10)

    calls = [
        (
            "/api/v1/tools/compare/models",
            {"model_ids": ["gpt-4o-mini", "claude-sonnet-5"]},
        ),
        (
            "/api/v1/tools/compare/vector-db",
            {"tool_slugs": ["pgvector", "qdrant"], "vector_count": 1_000_000},
        ),
        (
            "/api/v1/tools/compare/stacks",
            {"archetypes": ["serverless", "self-hosted"]},
        ),
        (
            "/api/v1/tools/compare/build-vs-buy",
            {"build_hours": 400, "vendor_monthly": 500},
        ),
    ]
    for path, payload in calls:
        response = await client.post(path, json=payload)
        assert response.status_code == 200, f"{path}: {response.text}"
        assert response.json()["data"]["source"] == "hybrid", path

    assert [purpose for purpose, _ in await _purposes(db)] == ["comparison_rationale"] * 4


# ── cost planner ─────────────────────────────────────────────────────────────


async def test_the_budget_estimator_adds_suggestions_without_dropping_the_costed_ones(
    client: AsyncClient, gemini: Any
) -> None:
    """The engine's rows carry a computed dollar figure; the model's carry a
    judgement about what the change costs. Replacing the first with the second
    would trade a number for an opinion."""
    gemini(
        {
            "cost_optimization": {
                "suggestions": [
                    {
                        "change": "Move the classification line to a smaller model.",
                        "saves": "$180/month",
                        "costs_you": "a few points of accuracy on edge cases",
                    }
                ]
            }
        }
    )

    response = await client.post(
        "/api/v1/tools/cost/budget-estimator",
        json={
            "lines": [
                {
                    "name": "chat",
                    "model_id": "gpt-4o-mini",
                    "requests_per_day": 5000,
                    "input_tokens": 4000,
                    "output_tokens": 400,
                }
            ],
            "monthly_growth_pct": 0.05,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]

    rows = data["tables"]["recommendations"]
    assert "ai_suggestion" in [row["kind"] for row in rows]
    suggestion = next(row for row in rows if row["kind"] == "ai_suggestion")
    assert "smaller model" in suggestion["detail"]
    assert suggestion["trade_off"] == "a few points of accuracy on edge cases"
    assert suggestion["monthly_saving"] == "$180/month"
    assert data["source"] == "hybrid"
    # The engine's costed rows survive, and every row has the new column so
    # the table does not render a half-empty one.
    assert [row["kind"] for row in rows if row["kind"] != "ai_suggestion"]
    assert all("trade_off" in row for row in rows)


async def test_a_suggestion_the_engine_already_made_annotates_it_instead_of_repeating_it(
    client: AsyncClient, gemini: Any
) -> None:
    """The engine computed the saving and cannot judge what the change costs
    in quality; the model is the other way round. Two rows saying the same
    thing is the worst of both — so the judgement lands on the costed row."""
    gemini(
        {
            "cost_optimization": {
                "suggestions": [
                    {
                        "change": "Cache the stable prompt prefix on the chat line at 80%.",
                        "saves": "$400/month",
                        "costs_you": "a cache invalidation story you do not have yet",
                    }
                ]
            }
        }
    )

    response = await client.post(
        "/api/v1/tools/cost/budget-estimator",
        json={
            "lines": [
                {
                    "name": "chat",
                    "model_id": "gpt-4o-mini",
                    "requests_per_day": 5000,
                    "input_tokens": 4000,
                    "output_tokens": 400,
                }
            ],
            "monthly_growth_pct": 0.05,
        },
    )
    data = response.json()["data"]

    rows = data["tables"]["recommendations"]
    assert "ai_suggestion" not in [row["kind"] for row in rows]
    cached = next(row for row in rows if row["kind"] == "caching")
    assert cached["trade_off"] == "a cache invalidation story you do not have yet"
    # The engine's computed figure is untouched by the annotation.
    assert cached["monthly_saving"]


# ── the architecture document ────────────────────────────────────────────────


async def _a_stack(client: AsyncClient, db: AsyncSession) -> str:
    """A signed-in Pro user with a saved stack, which is what an export needs."""
    from app.models.user import User
    from tests.conftest import GOOD_PASSWORD, register_and_verify

    email = "narrative@example.com"
    user_id = await register_and_verify(client, db, email=email)
    user = await db.get(User, user_id)
    assert user is not None
    user.plan = Plan.PRO
    await db.flush()

    login = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    token = login.json()["data"]["tokens"]["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"

    response = await client.post(
        "/api/v1/stacks",
        json={
            "name": "Client X RAG rollout",
            "component_slugs": ["anthropic-api", "llamaindex", "qdrant", "postgresql"],
            "requirements": {"use_case": "rag", "monthly_budget": 2000},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["data"]["id"])


DOCUMENT_REPLY = {
    "architecture_document": {
        "overview": "A managed-model RAG stack sized for one team.",
        "decisions": "Qdrant over pgvector because the corpus outgrows a shared Postgres.",
        "operations": "The reindex job is the thing that will page you.",
    }
}


async def test_an_exported_architecture_document_carries_written_sections(
    client: AsyncClient, db: AsyncSession, gemini: Any
) -> None:
    """The prompt for this has been in the registry since M16 and nothing
    called it, so every architecture document shipped as tables and headings
    with no prose in it at all."""
    gemini(DOCUMENT_REPLY)
    stack_id = await _a_stack(client, db)

    response = await client.post(
        "/api/v1/exports",
        json={
            "source_type": "stack",
            "source_id": stack_id,
            "format": "markdown",
            "artifact_type": "architecture",
        },
    )
    assert response.status_code == 201, response.text
    export_id = response.json()["data"]["id"]

    download = await client.get(f"/api/v1/exports/{export_id}/download")
    assert download.status_code == 200
    document = download.text

    assert "## Overview" in document
    assert "A managed-model RAG stack sized for one team." in document
    assert "## Design decisions" in document
    assert "## Operating this stack" in document
    # And the engine's half is still all there.
    assert "## Components" in document
    assert "Qdrant" in document


async def test_exports_with_nowhere_to_put_prose_do_not_pay_for_any(
    client: AsyncClient, db: AsyncSession, gemini: Any
) -> None:
    """A CSV of one table has no room for written sections, and a model call
    whose output is discarded is a cost that only shows up on the bill."""
    stub = gemini(DOCUMENT_REPLY)
    stack_id = await _a_stack(client, db)

    response = await client.post(
        "/api/v1/exports",
        json={
            "source_type": "stack",
            "source_id": stack_id,
            "format": "csv",
            "table": "components",
        },
    )
    assert response.status_code == 201, response.text
    assert stub.calls == []


async def test_a_failed_narration_still_exports_the_document(
    client: AsyncClient, db: AsyncSession, gemini: Any
) -> None:
    """An export is a paid feature. Failing one to protect prose would be the
    wrong trade — every figure in the document is the engine's own (D-06)."""
    gemini({})  # every call raises
    stack_id = await _a_stack(client, db)

    response = await client.post(
        "/api/v1/exports",
        json={
            "source_type": "stack",
            "source_id": stack_id,
            "format": "markdown",
            "artifact_type": "architecture",
        },
    )
    assert response.status_code == 201, response.text
    export_id = response.json()["data"]["id"]

    download = await client.get(f"/api/v1/exports/{export_id}/download")
    assert download.status_code == 200
    assert "## Components" in download.text
    assert "## Overview" not in download.text
