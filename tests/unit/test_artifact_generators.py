"""Artifact generators, against fixture sources (M18).

Every test here builds a source by hand and asserts on exact content. That is
possible because the generators take a resolved dataclass and touch no
database — the property `services/artifacts/sources.py` exists to protect.

The idempotency tests are the load-bearing ones. FR-11 requires that
re-exporting produces byte-identical output, and the only way to believe that
is to generate twice and compare bytes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.schemas.catalog import CompatibilityOut, CompatibilityPairOut, ToolOut
from app.schemas.tools import Artifact, Provenance, ToolRunOut, ToolWarning
from app.services import artifacts
from app.services.artifacts import (
    architecture,
    cursor_rules,
    deployment,
    result_document,
    roadmap,
    sources,
)
from app.services.artifacts import (
    markdown as md,
)
from app.services.artifacts.sources import RunSource, StackSource
from app.services.stack_score_service import score

STAMP = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


def tool(
    slug: str,
    *,
    name: str | None = None,
    category: str = "vector-db",
    status: str = "recommended",
    self_hostable: bool = True,
    license: str | None = "Apache-2.0",
    **facts: Any,
) -> ToolOut:
    base: dict[str, Any] = {
        "managed": True,
        "ops_burden": 2,
        "scale_ceiling": 4,
        "ecosystem": 4,
        "lock_in": 2,
        "free_tier": True,
    }
    base.update(facts)
    return ToolOut(
        id=f"tool_{slug}",
        slug=slug,
        name=name or slug.replace("-", " ").title(),
        category=category,
        description=f"{slug} description.",
        status=status,
        status_reason="Superseded." if status != "recommended" else None,
        maturity_score=85,
        license=license,
        self_hostable=self_hostable,
        pricing_model="open-core",
        docs_url=f"https://example.com/{slug}",
        facts=base,
        last_reviewed_at=STAMP,
    )


DEFAULT_COMPONENTS = [
    tool("anthropic-api", category="llm-provider", self_hostable=False, license="proprietary"),
    tool("llamaindex", category="rag-framework"),
    tool("qdrant", category="vector-db"),
    tool("postgresql", category="database"),
    tool("redis", category="cache"),
    tool("langfuse", category="observability"),
    tool("railway", category="deployment"),
]


def stack_source(
    components: list[ToolOut] | None = None,
    *,
    name: str = "Client X RAG rollout",
    requirements: dict[str, Any] | None = None,
    missing: list[str] | None = None,
) -> StackSource:
    chosen = DEFAULT_COMPONENTS if components is None else components
    resolved = sources.requirements_of(requirements or {})
    # A one-component stack has no pairs to score, matching what
    # `stack_source_of` does with a single-slug stack.
    compatibility = (
        CompatibilityOut(
            tools=[item.slug for item in chosen],
            pairs=[
                CompatibilityPairOut(
                    tool_a=chosen[0].name,
                    tool_b=chosen[1].name,
                    score=82,
                    dimensions={},
                    notes="Well trodden.",
                )
            ],
            overall=82,
        )
        if len(chosen) > 1
        else None
    )
    return StackSource(
        id="stk_fixture",
        title=name,
        description="Retrieval over the client's document set.",
        slug_basis=name,
        components=chosen,
        missing_slugs=missing or [],
        deprecated=[item for item in chosen if item.status not in {"recommended", "stable"}],
        requirements=resolved,
        score=score(
            chosen,
            monthly_budget=resolved.monthly_budget,
            scale_target=resolved.scale_target,
            sensitivity=resolved.sensitivity,
            compatibility=compatibility,
        ),
        compatibility=compatibility,
        version=3,
        updated_at=STAMP,
    )


def run_source(
    *,
    workflow: str = "roi",
    tool_slug: str = "hours-saved",
    emitted: list[Artifact] | None = None,
) -> RunSource:
    return RunSource(
        id="run_fixture",
        title="Hours Saved",
        slug_basis=tool_slug,
        tool_slug=tool_slug,
        workflow=workflow,
        input={"team_size": 12, "hours_per_week": 4},
        output=ToolRunOut(
            run_id="run_fixture",
            tool_slug=tool_slug,
            source="rule_based",
            duration_ms=7,
            created_at=STAMP,
            metrics={"monthly_hours": "192.00", "annual_value": "115200.00"},
            tables={"breakdown": [{"activity": "Triage", "hours_per_month": "96.00"}]},
            artifacts=emitted or [],
            warnings=[ToolWarning(level="warning", message="Adoption is modelled, not observed.")],
            provenance=Provenance(),
        ),
    )


# ── the nine P1 types ────────────────────────────────────────────────────────


def test_a_stack_produces_every_stack_scoped_p1_artifact() -> None:
    available = {descriptor.type for descriptor in artifacts.available(stack_source())}
    assert available == {
        "architecture",
        "diagram",
        "cost-estimate",
        "roadmap",
        "compose",
        "env",
        "cursor-rules",
    }


def test_an_roi_run_produces_a_business_case() -> None:
    available = {descriptor.type for descriptor in artifacts.available(run_source())}
    assert "business-case" in available


def test_an_emitted_artifact_is_returned_verbatim() -> None:
    """The file the user was shown is the file they get.

    Regenerating a compose file from the run's inputs would risk handing
    someone something different from what was on screen, and the difference
    would only surface when they ran it.
    """
    emitted = Artifact(
        type="compose",
        format="yaml",
        filename="docker-compose.yml",
        content="services:\n  app:\n    image: example\n",
    )
    source = run_source(workflow="infra", tool_slug="docker-compose", emitted=[emitted])

    assert artifacts.generate(source, "compose").content == emitted.content


def test_an_emitted_business_case_is_not_regenerated() -> None:
    emitted = Artifact(
        type="business-case",
        format="markdown",
        filename="business-case.md",
        content="# The one WF5 wrote\n",
    )
    source = run_source(emitted=[emitted])

    assert artifacts.generate(source, "business-case").content == "# The one WF5 wrote\n"


def test_an_unsupported_artifact_type_is_not_found() -> None:
    from app.core.errors import NotFound

    with pytest.raises(NotFound):
        artifacts.generate(run_source(), "cursor-rules")


# ── idempotency (FR-11) ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "artifact_type",
    ["architecture", "diagram", "cost-estimate", "roadmap", "compose", "env", "cursor-rules"],
)
def test_generating_twice_produces_identical_bytes(artifact_type: str) -> None:
    first = artifacts.generate(stack_source(), artifact_type)
    second = artifacts.generate(stack_source(), artifact_type)
    assert first.content.encode() == second.content.encode()


def test_the_whole_result_document_is_idempotent() -> None:
    assert result_document.render(run_source()) == result_document.render(run_source())


# ── the architecture document ────────────────────────────────────────────────


def test_the_architecture_document_carries_the_recomputed_score() -> None:
    source = stack_source()
    content = architecture.document(source).content

    assert f"{source.score.total}/100" in content
    # The breakdown, so the headline is checkable rather than asserted.
    assert "| Cost efficiency |" in content
    assert "Overall **82/100**" in content


def test_a_component_missing_from_the_catalog_is_reported_not_dropped() -> None:
    content = architecture.document(stack_source(missing=["ghost-db"])).content
    assert "no longer in the catalog" in content
    assert "`ghost-db`" in content


def test_the_diagram_only_draws_roles_the_stack_fills() -> None:
    source = stack_source(
        [
            tool("anthropic-api", category="llm-provider", self_hostable=False),
            tool("llamaindex", category="rag-framework"),
        ]
    )
    diagram = architecture.diagram(source).content

    assert "graph LR" in diagram
    assert "llm[" in diagram
    assert "vector_db[" not in diagram


# ── deployment ───────────────────────────────────────────────────────────────


def test_compose_uses_the_stacks_own_components() -> None:
    content = deployment.compose(stack_source()).content

    assert "qdrant/qdrant:latest" in content
    assert "postgres:17-alpine" in content
    assert "redis:7-alpine" in content
    # A managed provider gets a note, not an invented container.
    assert "managed service" in content


def test_two_components_mapping_to_one_container_emit_one_service() -> None:
    """`pgvector` and `postgresql` are the same Postgres.

    Emitting the service twice produces a file where the second definition
    silently wins, which is a corrupt file that still parses.
    """
    content = deployment.compose(
        stack_source(
            [
                tool("pgvector", category="vector-db"),
                tool("postgresql", category="database"),
            ]
        )
    ).content

    assert content.count("\n  postgres:") == 1


def test_a_component_with_no_image_says_so_rather_than_guessing() -> None:
    content = deployment.compose(stack_source([tool("faiss", category="vector-db")])).content
    assert "no container" in content
    assert "faiss:" not in content


def test_env_example_covers_every_variable_the_compose_file_reads() -> None:
    source = stack_source()
    compose = deployment.compose(source).content
    env = deployment.env_example(source).content

    for variable in ("POSTGRES_PASSWORD", "ANTHROPIC_API_KEY"):
        assert f"${{{variable}}}" in compose or variable in compose
        assert f"{variable}=" in env


def test_nothing_in_env_example_is_a_credential() -> None:
    content = deployment.env_example(stack_source()).content
    assert "Nothing in this file is a credential" in content
    for line in content.splitlines():
        if "=" in line and not line.startswith("#"):
            assert "change-me" in line or "change_me" in line or "your-" in line or "llama" in line


# ── roadmap and rules ────────────────────────────────────────────────────────


def test_the_roadmap_skips_roles_the_stack_does_not_fill() -> None:
    steps = roadmap.steps(
        stack_source(
            [
                tool("anthropic-api", category="llm-provider", self_hostable=False),
                tool("llamaindex", category="rag-framework"),
            ]
        )
    )
    titles = [step["title"] for step in steps]

    assert "Wire the model provider" in titles
    assert "Provision the vector store" not in titles
    # Every stack gets the closing phase.
    assert titles[-1] == "Evaluate against real inputs"


def test_cursor_rules_name_the_components_and_their_gotchas() -> None:
    content = cursor_rules.generate(stack_source()).content

    assert "**Qdrant**" in content
    assert "named vectors" in content
    assert "Do not invent pricing" in content


def test_cursor_rules_forbid_managed_apis_on_restricted_data() -> None:
    content = cursor_rules.generate(
        stack_source(requirements={"sensitivity": "restricted"})
    ).content
    assert "Data must not leave the network" in content


def test_a_deprecated_component_reaches_the_rules_file() -> None:
    content = cursor_rules.generate(
        stack_source(
            [
                tool("chroma", category="vector-db", status="caution"),
                tool("llamaindex", category="rag-framework"),
            ]
        )
    ).content
    assert "Do not build further on **Chroma**" in content


# ── the result document ──────────────────────────────────────────────────────


def test_the_result_document_keeps_warnings() -> None:
    """A shared plan that drops "this is deprecated" is the version that gets
    forwarded to whoever implements it."""
    content = result_document.render(run_source())
    assert "## Warnings" in content
    assert "Adoption is modelled" in content


def test_an_embedded_fenced_block_does_not_break_the_outer_fence() -> None:
    emitted = Artifact(
        type="notes",
        format="markdown",
        filename="notes.md",
        content="Here is a block:\n\n```yaml\nkey: value\n```\n",
    )
    content = result_document.render(run_source(emitted=[emitted]))

    assert "````" in content
    assert "key: value" in content


# ── markdown primitives ──────────────────────────────────────────────────────


def test_a_pipe_in_a_cell_cannot_split_the_row() -> None:
    rendered = md.table([{"name": "a|b", "note": "x"}])
    body = rendered.splitlines()[-1]
    assert body.count("|") == 3
    assert "a/b" in body
