"""Stack Architect, end to end against the real catalog (M15).

The constraint tests are the important ones. Each asserts that a hard
constraint *eliminated* a component rather than ranking it down — a
compliance-violating stack at rank three is still a stack a tired reader
picks, which is why "eliminate, do not penalise" is a rule rather than a
preference.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import GOOD_PASSWORD, register_and_verify

pytestmark = pytest.mark.usefixtures("seeded_catalog")

RECOMMEND = "/api/v1/architect/recommend"
SCORE = "/api/v1/architect/score"
STACKS = "/api/v1/stacks"


async def _recommend(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    response = await client.post(RECOMMEND, json=overrides)
    assert response.status_code == 200, response.text
    return response.json()["data"]


def _slugs(data: dict[str, Any]) -> set[str]:
    return {row["slug"] for row in data["tables"]["components"]}


def _excluded_by(data: dict[str, Any], constraint: str) -> set[str]:
    return {
        row["slug"]
        for row in data["tables"]["exclusions"]
        if row["constraint"] == constraint.replace("_", " ")
    }


def _diagram(data: dict[str, Any]) -> tuple[set[str], list[tuple[str, str]]]:
    """The diagram's declared nodes and its edges, solid and dotted alike.

    `-->` is three characters and `-.->` is four, so a class matching one
    character between the dashes silently sees only the dotted edges — which
    is every supporting role and no part of the request path.
    """
    source = next(a for a in data["artifacts"] if a["format"] == "mermaid")["content"]
    assert source.startswith("graph LR")
    return (
        set(re.findall(r"^\s{4}(\w+)\[", source, re.M)),
        re.findall(r"^\s{4}(\w+) -\.?-> (\w+)$", source, re.M),
    )


# ── the headline ─────────────────────────────────────────────────────────────


async def test_a_recommendation_returns_a_scored_stack_with_a_diagram(
    client: AsyncClient,
) -> None:
    data = await _recommend(client)

    assert float(data["metrics"]["score"]) > 0
    assert len(data["tables"]["components"]) >= 4
    assert len(data["tables"]["score_breakdown"]) == 10

    declared, edges = _diagram(data)
    assert edges
    assert all(source in declared and target in declared for source, target in edges)


async def test_every_node_in_the_diagram_is_connected(client: AsyncClient) -> None:
    """No orphans, on the shape that declares the most nodes.

    The diagram is rendered now rather than shown as source, and a role that
    declares a node but draws no edge is a box floating beside the picture.
    Self-hosting plus an agent use case is what turns on both of M25's
    optional roles at once — `restricted` also turns guardrails on, but it
    eliminates every GPU vendor on the way, so it never declares the compute
    node this is here to catch.
    """
    data = await _recommend(client, model_hosting="self-hosted", use_case="agents")

    declared, edges = _diagram(data)
    connected = {node for edge in edges for node in edge}

    assert {"compute", "guardrails"} <= declared
    assert declared - connected == set()
    assert connected - declared == set()


async def test_compute_hangs_off_the_model_not_the_framework(
    client: AsyncClient,
) -> None:
    """Where the weights run belongs to the model, not the orchestration glue."""
    data = await _recommend(client, model_hosting="self-hosted")

    _, edges = _diagram(data)

    assert ("llm", "compute") in edges
    assert ("framework", "compute") not in edges


async def test_guardrails_sits_between_the_framework_and_the_model(
    client: AsyncClient,
) -> None:
    """In the request path, not off to one side — it inspects the traffic.

    An agent use case turns guardrails on without self-hosting, which is the
    common way this role appears.
    """
    data = await _recommend(client, use_case="agents")

    _, edges = _diagram(data)

    assert ("framework", "guardrails") in edges
    assert ("guardrails", "llm") in edges
    assert ("framework", "llm") not in edges


async def test_a_stack_without_guardrails_keeps_its_framework_to_model_edge(
    client: AsyncClient,
) -> None:
    """The chain closes over the role it dropped."""
    data = await _recommend(client)

    declared, edges = _diagram(data)

    assert "guardrails" not in declared
    assert ("framework", "llm") in edges


async def test_the_score_breakdown_sums_to_the_headline(client: AsyncClient) -> None:
    """The number the whole screen is built around has to be checkable."""
    data = await _recommend(client)

    total = sum(float(row["contribution"]) for row in data["tables"]["score_breakdown"])
    assert abs(total - float(data["metrics"]["score"])) < 0.6


async def test_alternatives_name_what_they_trade(client: AsyncClient) -> None:
    data = await _recommend(client)
    alternatives = data["tables"]["alternatives"]

    assert alternatives
    for row in alternatives:
        assert row["strongest"] and row["weakest"]
        assert float(row["score"]) <= float(data["metrics"]["score"])


async def test_with_ai_disabled_it_still_returns_a_full_recommendation(
    client: AsyncClient,
) -> None:
    """The flagship degrades to a complete answer, never to an error page."""
    data = await _recommend(client)

    assert data["source"] == "rule_based"
    assert data["ai"] is None
    assert data["metrics"]["summary"]
    assert data["tables"]["components"]


# ── hard constraints eliminate ───────────────────────────────────────────────


async def test_restricted_data_eliminates_everything_that_cannot_self_host(
    client: AsyncClient,
) -> None:
    data = await _recommend(client, sensitivity="restricted")

    excluded = _excluded_by(data, "data_sensitivity")
    assert excluded
    # And nothing excluded came back in the recommendation.
    assert not (excluded & _slugs(data))
    for row in data["tables"]["components"]:
        assert row["self_hostable"] == "yes"


async def test_self_hosted_deployment_eliminates_api_only_providers(
    client: AsyncClient,
) -> None:
    data = await _recommend(client, deployment="self-hosted")

    assert _excluded_by(data, "deployment")
    for row in data["tables"]["components"]:
        assert row["self_hostable"] == "yes"


async def test_a_small_budget_eliminates_components_that_cost_more(
    client: AsyncClient,
) -> None:
    """A floor above the entire budget is not a low-scoring option.

    $50 rather than a rounder number because the catalog's published floors
    are $0, $25, and $95 — a threshold the data cannot discriminate on would
    make this test pass for the wrong reason.
    """
    generous = await _recommend(client, monthly_budget=50_000)
    tight = await _recommend(client, monthly_budget=50)

    excluded = _excluded_by(tight, "budget")
    assert excluded
    assert not (excluded & _slugs(tight))
    # Priced out of the tight run, available in the generous one.
    assert excluded - _excluded_by(generous, "budget")


async def test_a_tight_latency_budget_eliminates_batch_components(
    client: AsyncClient,
) -> None:
    data = await _recommend(client, latency_ms=200)

    assert _excluded_by(data, "latency")
    assert not (_excluded_by(data, "latency") & _slugs(data))


async def test_a_beginner_team_never_gets_a_component_that_needs_an_operator(
    client: AsyncClient,
) -> None:
    data = await _recommend(client, team_skill="beginner")

    assert _excluded_by(data, "team_skill")
    assert not (_excluded_by(data, "team_skill") & _slugs(data))


async def test_a_buried_tool_is_never_recommended(client: AsyncClient) -> None:
    """The graveyard exists so people stop reaching for these. Putting one in
    a fresh recommendation undoes that in a single screen."""
    data = await _recommend(client)

    for row in data["tables"]["components"]:
        assert row["status"] in {"recommended", "stable"}
    assert int(data["metrics"]["deprecated_components"]) == 0


async def test_every_exclusion_says_which_constraint_removed_it(
    client: AsyncClient,
) -> None:
    data = await _recommend(client, sensitivity="regulated", team_skill="beginner")

    for row in data["tables"]["exclusions"]:
        assert row["constraint"]
        assert len(row["reason"]) > 10


# ── scoring an explicit stack ────────────────────────────────────────────────


async def test_scoring_an_explicit_stack_flags_a_deprecated_component(
    client: AsyncClient,
) -> None:
    """FR-20: on the row, not in a footnote."""
    response = await client.post(
        SCORE, json={"component_slugs": ["pgvector", "autogpt"], "monthly_budget": 2_000}
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert int(data["metrics"]["deprecated_components"]) == 1
    assert any(w["level"] == "critical" for w in data["warnings"])


async def test_scoring_is_order_independent(client: AsyncClient) -> None:
    forward = await client.post(SCORE, json={"component_slugs": ["pgvector", "langgraph", "redis"]})
    reverse = await client.post(SCORE, json={"component_slugs": ["redis", "langgraph", "pgvector"]})

    assert forward.json()["data"]["metrics"]["score"] == reverse.json()["data"]["metrics"]["score"]
    assert (
        forward.json()["data"]["metrics"]["compatibility"]
        == reverse.json()["data"]["metrics"]["compatibility"]
    )


async def test_an_unknown_component_is_a_404(client: AsyncClient) -> None:
    response = await client.post(SCORE, json={"component_slugs": ["not-a-real-tool"]})
    assert response.status_code == 404


# ── saving, versioning, diffing ──────────────────────────────────────────────


async def _login(client: AsyncClient, db: AsyncSession, email: str) -> None:
    await register_and_verify(client, db, email=email)
    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    token = response.json()["data"]["tokens"]["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"


async def test_recommending_and_saving_both_need_a_session(anon_client: AsyncClient) -> None:
    """`/recommend` used to be the public demo — one full recommendation with
    no account, on the theory that the product working is the best pitch for
    it. It is behind the same door as everything else now."""
    assert (await anon_client.post(RECOMMEND, json={})).status_code == 401
    assert (
        await anon_client.post(STACKS, json={"name": "Mine", "component_slugs": ["pgvector"]})
    ).status_code == 401


async def test_save_version_and_diff(client: AsyncClient, db: AsyncSession) -> None:
    await _login(client, db, "architect@example.com")

    created = await client.post(
        STACKS,
        json={
            "name": "Support RAG",
            "component_slugs": ["pgvector", "langgraph"],
            "requirements": {"monthly_budget": 2_000, "scale_target": "medium"},
        },
    )
    assert created.status_code == 201
    stack = created.json()["data"]
    assert stack["current_version"] == 1
    assert float(stack["score"]) > 0

    updated = await client.patch(
        f"{STACKS}/{stack['id']}",
        json={
            "component_slugs": ["pgvector", "langgraph", "redis"],
            "change_summary": "Added a cache",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["current_version"] == 2

    versions = await client.get(f"{STACKS}/{stack['id']}/versions")
    assert [row["version"] for row in versions.json()["data"]] == [2, 1]

    diff = await client.get(f"{STACKS}/{stack['id']}/diff", params={"from": 1, "to": 2})
    assert diff.status_code == 200

    body = diff.json()["data"]
    assert body["added"] == ["redis"]
    assert body["removed"] == []
    assert sorted(body["unchanged"]) == ["langgraph", "pgvector"]
    assert body["change_summary"] == "Added a cache"
    # Both sides re-scored against today's catalog, so the delta is the edit.
    assert float(body["score_to"]) - float(body["score_from"]) == pytest.approx(
        float(body["score_delta"]), abs=0.05
    )


async def test_a_clone_starts_its_own_history(client: AsyncClient, db: AsyncSession) -> None:
    await _login(client, db, "cloner@example.com")

    stack = (
        await client.post(STACKS, json={"name": "Base", "component_slugs": ["pgvector"]})
    ).json()["data"]
    await client.patch(f"{STACKS}/{stack['id']}", json={"name": "Base v2"})

    clone = await client.post(f"{STACKS}/{stack['id']}/clone")
    assert clone.status_code == 201

    body = clone.json()["data"]
    assert body["id"] != stack["id"]
    assert body["name"] == "Base v2 (copy)"
    assert body["current_version"] == 1


async def test_a_stack_belongs_to_its_owner(client: AsyncClient, db: AsyncSession) -> None:
    await _login(client, db, "owner@example.com")
    stack = (
        await client.post(STACKS, json={"name": "Private", "component_slugs": ["pgvector"]})
    ).json()["data"]

    await _login(client, db, "stranger@example.com")
    # 404 rather than 403: distinguishing "not yours" from "does not exist"
    # tells an attacker which ids are real.
    assert (await client.get(f"{STACKS}/{stack['id']}")).status_code == 404


async def test_a_saved_stack_is_rescored_from_the_catalog_on_every_read(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The reason the score is not a column: a component buried after the save
    shows up the next time the stack is opened."""
    await _login(client, db, "rescore@example.com")

    stack = (
        await client.post(
            STACKS, json={"name": "Has a risk", "component_slugs": ["pgvector", "autogpt"]}
        )
    ).json()["data"]

    assert "autogpt" in stack["deprecated_components"]

    fetched = await client.get(f"{STACKS}/{stack['id']}")
    assert fetched.json()["data"]["deprecated_components"] == ["autogpt"]


async def test_diffing_a_version_against_itself_is_refused(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _login(client, db, "differ@example.com")
    stack = (
        await client.post(STACKS, json={"name": "One", "component_slugs": ["pgvector"]})
    ).json()["data"]

    response = await client.get(f"{STACKS}/{stack['id']}/diff", params={"from": 1, "to": 1})
    assert response.status_code == 422
