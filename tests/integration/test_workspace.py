"""Projects, saved runs, the dashboard, and search (M17).

The file's centre of gravity is `test_no_endpoint_returns_another_users_data`.
It is parametrised over every user-scoped endpoint, so an endpoint added later
without scoping fails this suite rather than leaking in production — which is
the failure mode that actually matters in this module.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.models.tool_run import ToolRun
from app.models.user import Plan, User
from app.services import run_service
from tests.conftest import GOOD_PASSWORD, register_and_verify

pytestmark = pytest.mark.usefixtures("seeded_catalog")

PROJECTS = "/api/v1/projects"
RUNS = "/api/v1/runs"
DASHBOARD = "/api/v1/dashboard"
SEARCH = "/api/v1/search"
LLM_PRICING = "/api/v1/tools/cost/llm-pricing"

BASE_RUN = {
    "model_id": "gpt-4o-mini",
    "input_tokens": 1000,
    "output_tokens": 500,
    "requests_per_day": 100,
}


async def _sign_in(client: AsyncClient, db: AsyncSession, email: str, *, plan: Plan = Plan.PRO):
    """Register, verify, upgrade, and attach the bearer token to the client."""
    user_id = await register_and_verify(client, db, email=email)

    user = await db.get(User, user_id)
    assert user is not None
    user.plan = plan
    await db.flush()

    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    token = response.json()["data"]["tokens"]["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return user


async def _a_run(client: AsyncClient) -> str:
    response = await client.post(LLM_PRICING, json=BASE_RUN)
    assert response.status_code == 200
    return response.json()["data"]["run_id"]


# ── the save model ───────────────────────────────────────────────────────────


async def test_every_run_is_logged_whether_or_not_it_is_saved(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "logger@example.com")
    run_id = await _a_run(client)

    listed = await client.get(RUNS)
    assert [row["id"] for row in listed.json()["data"]] == [run_id]
    assert listed.json()["data"][0]["saved"] is False


async def test_saving_and_unsaving_a_run(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "saver@example.com")
    run_id = await _a_run(client)

    saved = await client.post(f"{RUNS}/{run_id}/save")
    assert saved.status_code == 200
    assert saved.json()["data"]["saved"] is True

    only_saved = await client.get(RUNS, params={"saved": True})
    assert [row["id"] for row in only_saved.json()["data"]] == [run_id]

    unsaved = await client.delete(f"{RUNS}/{run_id}/save")
    assert unsaved.json()["data"]["saved"] is False
    # The row survives — it is still real history.
    assert (await client.get(f"{RUNS}/{run_id}")).status_code == 200


async def test_an_anonymous_user_cannot_save_but_can_still_run(client: AsyncClient) -> None:
    """Running and exporting are free. Keeping is what the account is for."""
    run_id = await _a_run(client)

    response = await client.post(f"{RUNS}/{run_id}/save")
    assert response.status_code == 403


async def test_the_purge_removes_unsaved_runs_and_spares_saved_ones(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The one durability promise the product makes about a run."""
    await _sign_in(client, db, "purge@example.com")
    keep_id = await _a_run(client)
    drop_id = await _a_run(client)
    await client.post(f"{RUNS}/{keep_id}/save")

    old = utcnow() - timedelta(days=run_service.RETENTION_DAYS + 1)
    for run_id in (keep_id, drop_id):
        run = await db.get(ToolRun, run_id)
        assert run is not None
        run.created_at = old
    await db.flush()

    removed = await run_service.purge_expired(db)
    assert removed == 1

    surviving = (await db.execute(select(ToolRun.id))).scalars().all()
    assert keep_id in surviving
    assert drop_id not in surviving


async def test_a_recent_unsaved_run_is_not_purged(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "recent@example.com")
    run_id = await _a_run(client)

    assert await run_service.purge_expired(db) == 0
    assert await db.get(ToolRun, run_id) is not None


async def test_an_anonymous_run_is_claimed_on_signup(client: AsyncClient, db: AsyncSession) -> None:
    """The highest-intent conversion moment the product has: the work is
    already done, and the account is what keeps it."""
    run_id = await _a_run(client)

    anonymous = await db.get(ToolRun, run_id)
    assert anonymous is not None
    assert anonymous.user_id is None
    assert anonymous.anonymous_session_id is not None

    await _sign_in(client, db, "claimer@example.com")

    claimed = await db.get(ToolRun, run_id)
    await db.refresh(claimed)  # type: ignore[arg-type]
    assert claimed is not None
    assert claimed.user_id is not None
    assert claimed.anonymous_session_id is None

    dashboard = await client.get(DASHBOARD)
    assert run_id in {row["run_id"] for row in dashboard.json()["data"]["recent_runs"]}


# ── projects ─────────────────────────────────────────────────────────────────


async def test_project_crud_and_items(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "projects@example.com")
    run_id = await _a_run(client)

    created = await client.post(PROJECTS, json={"name": "Client X rollout", "use_case": "rag"})
    assert created.status_code == 201
    project = created.json()["data"]

    added = await client.post(
        f"{PROJECTS}/{project['id']}/items",
        json={"item_type": "run", "item_id": run_id, "note": "The baseline"},
    )
    assert added.status_code == 201
    assert added.json()["data"]["title"] == "llm-pricing"
    assert added.json()["data"]["href"].startswith("/cost?run=")

    listed = await client.get(f"{PROJECTS}/{project['id']}/items")
    assert len(listed.json()["data"]) == 1

    fetched = await client.get(f"{PROJECTS}/{project['id']}")
    assert fetched.json()["data"]["item_count"] == 1

    removed = await client.delete(f"{PROJECTS}/{project['id']}/items/{added.json()['data']['id']}")
    assert removed.status_code == 204
    assert (await client.get(f"{PROJECTS}/{project['id']}/items")).json()["data"] == []


async def test_items_reorder_and_pin(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "order@example.com")
    project = (await client.post(PROJECTS, json={"name": "Ordering"})).json()["data"]

    ids = []
    for _ in range(3):
        run_id = await _a_run(client)
        item = await client.post(
            f"{PROJECTS}/{project['id']}/items", json={"item_type": "run", "item_id": run_id}
        )
        ids.append(item.json()["data"]["id"])

    reversed_ids = list(reversed(ids))
    reordered = await client.patch(
        f"{PROJECTS}/{project['id']}/items/order", json={"item_ids": reversed_ids}
    )
    assert [row["id"] for row in reordered.json()["data"]] == reversed_ids

    pinned = await client.patch(
        f"{PROJECTS}/{project['id']}/items/{ids[0]}/pin", json={"pinned": True}
    )
    assert pinned.json()["data"]["pinned"] is True

    # Pinned first, then the arrangement.
    listed = await client.get(f"{PROJECTS}/{project['id']}/items")
    assert listed.json()["data"][0]["id"] == ids[0]


async def test_a_partial_reorder_keeps_the_omitted_items(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A client reordering a filtered view must not silently drop what it
    could not see."""
    await _sign_in(client, db, "partial@example.com")
    project = (await client.post(PROJECTS, json={"name": "Partial"})).json()["data"]

    ids = []
    for _ in range(3):
        run_id = await _a_run(client)
        item = await client.post(
            f"{PROJECTS}/{project['id']}/items", json={"item_type": "run", "item_id": run_id}
        )
        ids.append(item.json()["data"]["id"])

    result = await client.patch(
        f"{PROJECTS}/{project['id']}/items/order", json={"item_ids": [ids[2]]}
    )
    returned = [row["id"] for row in result.json()["data"]]

    assert returned[0] == ids[2]
    assert set(returned) == set(ids)


async def test_deleting_a_project_keeps_the_work_it_held(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The container is the arrangement. Deleting it must not be a way to lose
    work the user did not intend to delete."""
    await _sign_in(client, db, "container@example.com")
    run_id = await _a_run(client)
    project = (await client.post(PROJECTS, json={"name": "Temporary"})).json()["data"]
    await client.post(
        f"{PROJECTS}/{project['id']}/items", json={"item_type": "run", "item_id": run_id}
    )

    assert (await client.delete(f"{PROJECTS}/{project['id']}")).status_code == 204
    assert (await client.get(f"{RUNS}/{run_id}")).status_code == 200


async def test_the_free_plan_has_no_projects(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "free@example.com", plan=Plan.FREE)

    response = await client.post(PROJECTS, json={"name": "Not allowed"})
    assert response.status_code == 402
    assert response.json()["error"]["details"]["quota"]["metric"] == "projects"


async def test_the_project_limit_is_enforced_per_plan(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Also asserts M20's promise that a limit changes without a deploy: the
    cap is lowered by writing the row, not by patching a constant."""
    from app.models.billing import Metric
    from tests.conftest import set_limit

    await set_limit(db, plan=Plan.PRO, metric=Metric.PROJECTS, value=2)
    await _sign_in(client, db, "limited@example.com")

    for index in range(2):
        assert (await client.post(PROJECTS, json={"name": f"P{index}"})).status_code == 201

    over = await client.post(PROJECTS, json={"name": "One too many"})
    assert over.status_code == 402
    assert over.json()["error"]["details"]["quota"]["limit"] == 2


async def test_an_item_the_user_does_not_own_cannot_be_attached(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Otherwise a project is a way to read any run id by adding it."""
    await _sign_in(client, db, "victim2@example.com")
    victim_run = await _a_run(client)

    await _sign_in(client, db, "thief2@example.com")
    project = (await client.post(PROJECTS, json={"name": "Mine"})).json()["data"]

    response = await client.post(
        f"{PROJECTS}/{project['id']}/items", json={"item_type": "run", "item_id": victim_run}
    )
    assert response.status_code == 404


async def test_an_unbuilt_item_type_is_refused_rather_than_dangling(
    client: AsyncClient, db: AsyncSession
) -> None:
    """`template` is the remaining unbuilt type — M19 gives it a table.

    `artifact` was here too until M18 gave artifacts one; the refusal is about
    what can be *resolved*, so the list shrinks as tables arrive rather than
    staying as a permanent denial.
    """
    await _sign_in(client, db, "templates@example.com")
    project = (await client.post(PROJECTS, json={"name": "Future"})).json()["data"]

    response = await client.post(
        f"{PROJECTS}/{project['id']}/items",
        json={"item_type": "template", "item_id": "tpl_whatever"},
    )
    assert response.status_code == 422


# ── the carried session ──────────────────────────────────────────────────────


async def test_the_session_merges_rather_than_replaces(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A tool handing two figures forward must not wipe the six a previous
    tool contributed."""
    await _sign_in(client, db, "session@example.com")
    project = (await client.post(PROJECTS, json={"name": "Carried"})).json()["data"]
    url = f"{PROJECTS}/{project['id']}/session"

    await client.patch(
        url,
        json={
            "values": {
                "monthly_llm_cost": "126.00",
                "llm_model": "claude-sonnet-5",
                "carried_from": [{"tool": "llm-pricing", "run_id": "run_1"}],
            }
        },
    )
    second = await client.patch(
        url,
        json={
            "values": {
                "vector_db": "qdrant",
                "carried_from": [{"tool": "rag-architecture", "run_id": "run_2"}],
            }
        },
    )

    state = second.json()["data"]["session_state"]
    assert state["monthly_llm_cost"] == "126.00"
    assert state["vector_db"] == "qdrant"
    # Provenance accumulates, so the UI can say where each number came from.
    assert [entry["tool"] for entry in state["carried_from"]] == [
        "llm-pricing",
        "rag-architecture",
    ]


async def test_the_session_survives_a_reread(client: AsyncClient, db: AsyncSession) -> None:
    """JSONB mutated in place is silently dropped on flush. This is the test
    that would catch it."""
    await _sign_in(client, db, "persist@example.com")
    project = (await client.post(PROJECTS, json={"name": "Persisted"})).json()["data"]

    await client.patch(
        f"{PROJECTS}/{project['id']}/session", json={"values": {"budget_monthly": "1200.00"}}
    )
    fetched = await client.get(f"{PROJECTS}/{project['id']}/session")

    assert fetched.json()["data"]["session_state"]["budget_monthly"] == "1200.00"


# ── dashboard and search ─────────────────────────────────────────────────────


async def test_the_dashboard_aggregates_match_the_underlying_rows(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "dash@example.com")
    first = await _a_run(client)
    await _a_run(client)
    await client.post(f"{RUNS}/{first}/save")
    await client.post(PROJECTS, json={"name": "Dash project"})

    data = (await client.get(DASHBOARD)).json()["data"]

    assert data["usage"]["total"] == 2
    assert data["usage"]["saved"] == 1
    assert data["usage"]["today"] == 2
    assert data["usage"]["projects"] == 1
    assert len(data["recent_runs"]) == 2
    assert len(data["projects"]) == 1
    assert data["plan"]["plan"] == "pro"
    # A feed row without a figure is a timestamp with extra steps.
    assert data["recent_runs"][0]["headline"]["value"]


async def test_the_stale_panel_surfaces_a_deprecated_tool_in_a_saved_stack(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The retention mechanic as a surface rather than an email."""
    await _sign_in(client, db, "stale@example.com")
    await client.post(
        "/api/v1/stacks",
        json={"name": "Aging stack", "component_slugs": ["pgvector", "autogpt"]},
    )

    data = (await client.get(DASHBOARD)).json()["data"]
    alerts = data["stale_alerts"]

    assert len(alerts) == 1
    assert alerts[0]["stack_name"] == "Aging stack"
    assert alerts[0]["reason"]
    assert data["saved_stacks"][0]["deprecated"] == ["autogpt"]


async def test_quick_start_reflects_what_this_user_actually_runs(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "quick@example.com")
    await _a_run(client)

    data = (await client.get(DASHBOARD)).json()["data"]
    assert data["quick_start"] == ["llm-pricing"]


async def test_a_new_account_gets_empty_panels_not_invented_ones(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "brandnew@example.com")

    data = (await client.get(DASHBOARD)).json()["data"]
    assert data["recent_runs"] == []
    assert data["quick_start"] == []
    assert data["stale_alerts"] == []
    assert data["usage"]["total"] == 0


async def test_search_finds_a_project_a_stack_and_a_run(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "search@example.com")
    run_id = await _a_run(client)
    await client.post(f"{RUNS}/{run_id}/save")
    await client.post(PROJECTS, json={"name": "Falcon migration", "description": "RAG rollout"})
    await client.post(
        "/api/v1/stacks", json={"name": "Falcon stack", "component_slugs": ["qdrant", "langgraph"]}
    )

    by_name = (await client.get(SEARCH, params={"q": "Falcon"})).json()["data"]
    assert {hit["kind"] for hit in by_name} == {"project", "stack"}

    by_tool = (await client.get(SEARCH, params={"q": "llm-pricing"})).json()["data"]
    assert [hit["kind"] for hit in by_tool] == ["run"]

    # A stack is findable by what is in it, not only by what it was named.
    by_component = (await client.get(SEARCH, params={"q": "qdrant"})).json()["data"]
    assert [hit["title"] for hit in by_component] == ["Falcon stack"]


async def test_search_excludes_other_users_work(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "owner3@example.com")
    await client.post(PROJECTS, json={"name": "Zeppelin secret"})

    await _sign_in(client, db, "stranger3@example.com")
    hits = (await client.get(SEARCH, params={"q": "Zeppelin"})).json()["data"]

    assert hits == []


# ── the ownership matrix ─────────────────────────────────────────────────────


async def _fixtures_for(client: AsyncClient, db: AsyncSession, email: str) -> dict[str, str]:
    await _sign_in(client, db, email)
    run_id = await _a_run(client)
    await client.post(f"{RUNS}/{run_id}/save")
    project = (await client.post(PROJECTS, json={"name": f"{email} project"})).json()["data"]
    item = await client.post(
        f"{PROJECTS}/{project['id']}/items", json={"item_type": "run", "item_id": run_id}
    )
    stack = (
        await client.post(
            "/api/v1/stacks", json={"name": f"{email} stack", "component_slugs": ["pgvector"]}
        )
    ).json()["data"]
    return {
        "run": run_id,
        "project": project["id"],
        "item": item.json()["data"]["id"],
        "stack": stack["id"],
    }


def _endpoints(ids: dict[str, str]) -> list[tuple[str, str, dict[str, Any] | None]]:
    """Every user-scoped endpoint, as (method, path, body).

    Parametrised deliberately: an endpoint added later without scoping fails
    this test rather than leaking in production.
    """
    project, run, item, stack = ids["project"], ids["run"], ids["item"], ids["stack"]
    return [
        ("GET", f"{PROJECTS}/{project}", None),
        ("PATCH", f"{PROJECTS}/{project}", {"name": "hijacked"}),
        ("DELETE", f"{PROJECTS}/{project}", None),
        ("GET", f"{PROJECTS}/{project}/items", None),
        ("POST", f"{PROJECTS}/{project}/items", {"item_type": "run", "item_id": run}),
        ("DELETE", f"{PROJECTS}/{project}/items/{item}", None),
        ("PATCH", f"{PROJECTS}/{project}/items/order", {"item_ids": [item]}),
        ("PATCH", f"{PROJECTS}/{project}/items/{item}/pin", {"pinned": True}),
        ("GET", f"{PROJECTS}/{project}/session", None),
        ("PATCH", f"{PROJECTS}/{project}/session", {"values": {"x": "1"}}),
        ("DELETE", f"{PROJECTS}/{project}/session", None),
        ("GET", f"{RUNS}/{run}", None),
        ("POST", f"{RUNS}/{run}/save", None),
        ("DELETE", f"{RUNS}/{run}/save", None),
        ("DELETE", f"{RUNS}/{run}", None),
        ("GET", f"/api/v1/stacks/{stack}", None),
        ("PATCH", f"/api/v1/stacks/{stack}", {"name": "hijacked"}),
        ("DELETE", f"/api/v1/stacks/{stack}", None),
        ("POST", f"/api/v1/stacks/{stack}/clone", None),
        ("GET", f"/api/v1/stacks/{stack}/versions", None),
        ("GET", f"/api/v1/stacks/{stack}/versions/1", None),
        ("GET", f"/api/v1/stacks/{stack}/diff?from=1&to=2", None),
    ]


async def test_no_endpoint_returns_another_users_data(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The security guarantee of this module, proven per resource type.

    404 rather than 403 throughout: distinguishing "not yours" from "does not
    exist" turns every id-taking endpoint into an oracle for which ids are
    real.
    """
    owned = await _fixtures_for(client, db, "alice@example.com")
    await _sign_in(client, db, "mallory@example.com")

    leaks: list[str] = []
    for method, path, body in _endpoints(owned):
        response = await client.request(method, path, json=body)
        if response.status_code != 404:
            leaks.append(f"{method} {path} -> {response.status_code}")

    assert leaks == [], "these endpoints did not scope to the caller: " + "; ".join(leaks)


async def test_listing_endpoints_return_only_the_callers_rows(
    client: AsyncClient, db: AsyncSession
) -> None:
    owned = await _fixtures_for(client, db, "listowner@example.com")
    await _sign_in(client, db, "listmallory@example.com")

    assert (await client.get(PROJECTS)).json()["data"] == []
    assert (await client.get("/api/v1/stacks")).json()["data"] == []
    assert owned["run"] not in {row["id"] for row in (await client.get(RUNS)).json()["data"]}

    dashboard = (await client.get(DASHBOARD)).json()["data"]
    assert dashboard["recent_runs"] == []
    assert dashboard["saved_stacks"] == []
