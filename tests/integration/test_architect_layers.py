"""The compute and guardrails layers, end to end against the real catalog (M25).

The first test is the one that matters most. Both roles are optional, and the
whole design rests on their defaults making them disappear — a module that
widened the flagship for everyone in order to serve the minority who self-host
would have made the common answer worse. Every other test here is only
interesting if that one holds.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.usefixtures("seeded_catalog")

RECOMMEND = "/api/v1/architect/recommend"


async def _recommend(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    response = await client.post(RECOMMEND, json=overrides)
    assert response.status_code == 200, response.text
    return response.json()["data"]


def _roles(data: dict[str, Any]) -> set[str]:
    return {row["role_key"] for row in data["tables"]["components"]}


def _names(data: dict[str, Any], role: str) -> list[str]:
    return [row["name"] for row in data["tables"]["components"] if row["role_key"] == role]


def _removed_by(data: dict[str, Any], constraint: str) -> set[str]:
    return {
        row["slug"]
        for row in data["tables"]["exclusions"]
        if row["constraint"] == constraint.replace("_", " ")
    }


# ── the default is unchanged ─────────────────────────────────────────────────


async def test_the_default_form_returns_the_same_eight_roles_it_always_did(
    client: AsyncClient,
) -> None:
    """M25 defaults to the answers that make both new roles vanish. If this
    breaks, every saved stack and every M15 fixture is wrong."""
    data = await _recommend(client)

    assert "compute" not in _roles(data)
    assert "guardrails" not in _roles(data)


async def test_an_api_only_stack_does_not_list_gpu_vendors_as_exclusions(
    client: AsyncClient,
) -> None:
    """The exclusions table explains why a tool the user *expected* is missing.
    Someone who said they are calling an API did not expect six GPU vendors,
    and listing them would bury the constraints that actually bit."""
    data = await _recommend(client, model_hosting="api")

    excluded = {row["slug"] for row in data["tables"]["exclusions"]}
    assert not excluded & {"aws-gpu", "gcp-gpu", "runpod", "vast-ai"}


# ── the compute layer ────────────────────────────────────────────────────────


async def test_self_hosting_the_weights_adds_a_compute_row_and_a_runtime(
    client: AsyncClient,
) -> None:
    """Both halves, or neither is useful: a GPU cluster recommended next to
    OpenAI reads as an engine that did not understand the question."""
    data = await _recommend(client, model_hosting="self-hosted")

    assert "compute" in _roles(data)
    assert _names(data, "compute")

    llm = _names(data, "llm")
    assert llm == ["vLLM"], llm
    assert "openai-api" in _removed_by(data, "model_hosting")


async def test_managed_open_weights_rents_no_machine_and_picks_neither_extreme(
    client: AsyncClient,
) -> None:
    """Open weights someone else runs. Ollama is open weights you run, and
    OpenAI is someone else running its own — the answer is neither."""
    data = await _recommend(client, model_hosting="managed-open-weights")

    assert "compute" not in _roles(data)
    assert _names(data, "llm") == ["Together AI"]
    removed = _removed_by(data, "model_hosting")
    assert {"openai-api", "ollama"} <= removed


async def test_spiky_traffic_takes_the_vendor_that_stops_billing(
    client: AsyncClient,
) -> None:
    data = await _recommend(client, model_hosting="self-hosted", traffic="spiky")

    assert _names(data, "compute") == ["RunPod"]
    assert {"aws-gpu", "gcp-gpu", "azure-gpu", "lambda-labs"} <= _removed_by(data, "traffic")


async def test_training_needs_a_box_of_cards_not_a_single_rented_one(
    client: AsyncClient,
) -> None:
    """Both figures come from `gpu_pricing` at seed time, so this asserts the
    registry and the filter agree rather than asserting a hand-typed tier."""
    data = await _recommend(client, model_hosting="self-hosted", workload="training")

    assert "compute" in _roles(data)
    # One card each, whatever their VRAM.
    assert {"runpod", "vast-ai"} <= _removed_by(data, "workload")


async def test_renting_a_gpu_counts_as_self_hosting(client: AsyncClient) -> None:
    """`self_hostable` asks whether software can run on infrastructure the user
    controls. It is not a question about a vendor who rents infrastructure, and
    reading it as one removed the compute layer from the one request that most
    obviously wanted it."""
    data = await _recommend(client, model_hosting="self-hosted", deployment="self-hosted")

    assert "compute" in _roles(data)
    assert not _removed_by(data, "deployment") & {"aws-gpu", "runpod"}


async def test_regulated_data_removes_the_rented_machine_and_says_so(
    client: AsyncClient,
) -> None:
    """Someone else's datacentre is still someone else's. This is the
    constraint that should take the compute layer away, and it explains
    itself where the deployment rule would not have."""
    data = await _recommend(client, model_hosting="self-hosted", sensitivity="regulated")

    assert "compute" not in _roles(data)
    assert {"aws-gpu", "runpod"} <= _removed_by(data, "data_sensitivity")


async def test_a_compute_row_carries_a_real_instance_for_the_cost_handoff(
    client: AsyncClient,
) -> None:
    """The result page names the vendor and refuses to invent its bill (D-16),
    so it hands off to the tool that models utilisation and spot instead. That
    handoff is only worth offering if it opens on a machine that exists."""
    data = await _recommend(client, model_hosting="self-hosted")
    slug = data["metrics"]["compute_gpu"]

    gpus = await client.get("/api/v1/catalog/gpus")
    assert gpus.status_code == 200, gpus.text
    assert slug in {row["slug"] for row in gpus.json()["data"]}


async def test_a_stack_with_no_compute_layer_offers_no_instance(
    client: AsyncClient,
) -> None:
    data = await _recommend(client)
    assert "compute_gpu" not in data["metrics"]


# ── guardrails ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_case": "agents"},
        {"sensitivity": "confidential"},
        {"sensitivity": "regulated"},
    ],
)
async def test_guardrails_appear_where_they_are_warranted(
    client: AsyncClient, overrides: dict[str, Any]
) -> None:
    data = await _recommend(client, **overrides)
    assert "guardrails" in _roles(data)


async def test_guardrails_stay_out_of_an_ordinary_internal_rag_stack(
    client: AsyncClient,
) -> None:
    data = await _recommend(client, use_case="rag", sensitivity="internal")
    assert "guardrails" not in _roles(data)


# ── residency ────────────────────────────────────────────────────────────────


async def test_a_regional_requirement_removes_managed_tools_with_nothing_on_file(
    client: AsyncClient,
) -> None:
    """Failing closed is the only safe direction: a wrong `eu` claim is a
    compliance problem the product caused, and a silent pass is that claim
    made without saying it."""
    data = await _recommend(client, residency="eu")

    removed = _removed_by(data, "residency")
    assert "pinecone" in removed
    assert "openai-api" in removed

    reasons = [row["reason"] for row in data["tables"]["exclusions"] if row["slug"] == "pinecone"]
    assert reasons and "residency" in reasons[0].lower()


async def test_a_regional_requirement_still_returns_a_whole_stack(
    client: AsyncClient,
) -> None:
    """Self-hostable software is unconstrained, because it runs wherever the
    user runs it — so an EU answer is a self-hosted answer rather than a dead
    end."""
    data = await _recommend(client, residency="eu")

    assert data["tables"]["components"]
    assert float(data["metrics"]["score"]) > 0
    for row in data["tables"]["components"]:
        assert row["self_hostable"] == "yes", row["name"]


async def test_residency_any_is_not_a_filter(client: AsyncClient) -> None:
    default = await _recommend(client)
    explicit = await _recommend(client, residency="any")

    assert _names(explicit, "llm") == _names(default, "llm")
    assert not _removed_by(explicit, "residency")
