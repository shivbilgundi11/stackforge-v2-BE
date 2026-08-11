"""The template library over HTTP (M19).

Two properties carry this file.

**Everything reads without an account.** The library is the product's organic
acquisition surface, and an endpoint behind a token cannot be crawled or
shared. Every test here that does not say otherwise runs anonymous.

**Premium is previewed, not hidden.** A gated template still appears in every
listing, still carries its title and summary, and still serves the opening of
its body — because hiding the row loses the indexable page, which is the
channel that justifies half the module.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.template import Template
from app.models.user import Plan, User
from tests.conftest import GOOD_PASSWORD, register_and_verify

TEMPLATES = "/api/v1/templates"


@pytest.fixture(autouse=True)
async def seeded_templates(db: AsyncSession) -> None:
    """Seeded per test, inside the rolled-back transaction.

    Not folded into the session-scoped catalog fixture: these tests mutate
    counters and one of them edits a row, and a session-scoped commit would
    carry that into every later test.
    """
    from app.services.seed_service import SeedReport, _seed_templates

    await _seed_templates(db, SeedReport(), refresh=False)
    await db.flush()


async def _sign_in(client: AsyncClient, db: AsyncSession, email: str, *, plan: Plan) -> User:
    user_id = await register_and_verify(client, db, email=email)
    user = await db.get(User, user_id)
    assert user is not None
    user.plan = plan
    await db.flush()

    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    client.headers["Authorization"] = f"Bearer {response.json()['data']['tokens']['access_token']}"
    return user


# ── the hub ──────────────────────────────────────────────────────────────────


async def test_the_hub_renders_without_an_account(client: AsyncClient) -> None:
    response = await client.get(TEMPLATES)
    assert response.status_code == 200
    data = response.json()["data"]

    assert data["total"] == 30
    assert [category["key"] for category in data["categories"]] == [
        "stack",
        "blueprint",
        "code-starter",
        "prompt",
        "config",
        "checklist",
        "business",
    ]
    assert {category["key"]: category["count"] for category in data["categories"]} == {
        "stack": 5,
        "blueprint": 5,
        "code-starter": 4,
        "prompt": 5,
        "config": 4,
        "checklist": 4,
        "business": 3,
    }
    assert len(data["featured"]) == 4
    assert len(data["recent"]) == 6


async def test_facets_are_read_from_the_data(client: AsyncClient) -> None:
    """A filter that offers a value returning nothing reads as broken search."""
    data = (await client.get(f"{TEMPLATES}/facets")).json()["data"]

    assert "rag" in data["use_cases"]
    assert data["difficulties"] == ["beginner", "intermediate", "advanced"]
    assert len(data["tags"]) > 10


# ── search ───────────────────────────────────────────────────────────────────


async def _slugs(client: AsyncClient, **params: Any) -> list[str]:
    response = await client.get(f"{TEMPLATES}/list", params=params)
    assert response.status_code == 200, response.text
    return [row["slug"] for row in response.json()["data"]]


async def test_search_finds_a_template_by_a_term_only_in_its_body(
    client: AsyncClient,
) -> None:
    """The point of indexing the body. "Barge-in" appears nowhere in the Voice
    AI blueprint's title or summary."""
    assert "voice-ai-architecture" in await _slugs(client, q="barge-in")


async def test_a_title_match_outranks_a_body_mention(client: AsyncClient) -> None:
    """The reason the index is weighted. Unweighted, the longest document that
    happens to mention the term wins."""
    slugs = await _slugs(client, q="cursor rules")
    assert slugs[0] == "cursor-rules"


async def test_search_returns_nothing_for_a_term_in_no_template(
    client: AsyncClient,
) -> None:
    assert await _slugs(client, q="zzzznotarealword") == []


async def test_punctuation_in_a_query_does_not_error(client: AsyncClient) -> None:
    """`to_tsquery` raises on this; `websearch_to_tsquery` does not, which is
    why the query uses it."""
    response = await client.get(f"{TEMPLATES}/list", params={"q": "rag & (chat |"})
    assert response.status_code == 200


# ── filters ──────────────────────────────────────────────────────────────────


async def test_each_filter_narrows_the_result_set(client: AsyncClient) -> None:
    everything = await _slugs(client)
    assert len(everything) == 30

    by_category = await _slugs(client, category="checklist")
    assert len(by_category) == 4
    assert set(by_category) < set(everything)

    by_use_case = await _slugs(client, use_case="agents")
    assert 0 < len(by_use_case) < 30

    by_difficulty = await _slugs(client, difficulty="beginner")
    assert 0 < len(by_difficulty) < 30

    by_tag = await _slugs(client, tag="rag")
    assert 0 < len(by_tag) < 30


async def test_filters_compose(client: AsyncClient) -> None:
    both = await _slugs(client, category="stack", difficulty="advanced")
    assert set(both) <= set(await _slugs(client, category="stack"))
    assert set(both) <= set(await _slugs(client, difficulty="advanced"))


async def test_the_premium_filter_finds_what_free_can_use_today(
    client: AsyncClient,
) -> None:
    free_only = await _slugs(client, premium=False)
    premium_only = await _slugs(client, premium=True)

    assert len(free_only) == 23
    assert len(premium_only) == 7
    assert not set(free_only) & set(premium_only)


# ── the gate ─────────────────────────────────────────────────────────────────


PREMIUM_SLUG = "fastapi-rag"


async def test_a_premium_template_is_listed_not_hidden(client: AsyncClient) -> None:
    """Hiding the row loses the indexable page, which is the acquisition
    channel that justifies half the module."""
    assert PREMIUM_SLUG in await _slugs(client)


async def test_an_anonymous_visitor_gets_a_preview_and_no_files(
    client: AsyncClient, db: AsyncSession
) -> None:
    data = (await client.get(f"{TEMPLATES}/{PREMIUM_SLUG}")).json()["data"]

    assert data["locked"] is True
    assert data["truncated"] is True
    assert data["files"] == []
    assert data["title"] and data["summary"]

    full = await db.scalar(select(Template).where(Template.slug == PREMIUM_SLUG))
    assert full is not None
    assert len(data["content_markdown"]) < len(full.content_markdown)
    # Enough to judge whether the rest is worth paying for.
    assert len(data["content_markdown"]) > 200


async def test_the_preview_stops_on_a_paragraph_boundary(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A cut mid-sentence reads as a rendering bug, and the first thing a
    paywall communicates should not be that the page is broken.

    Asserted structurally rather than by inspecting the last character: a
    preview may legitimately end on a heading, a table row, or a fence, none of
    which end in punctuation. What must hold is that the cut point is a real
    paragraph break in the source.
    """
    data = (await client.get(f"{TEMPLATES}/{PREMIUM_SLUG}")).json()["data"]
    preview = data["content_markdown"]

    full = await db.scalar(select(Template).where(Template.slug == PREMIUM_SLUG))
    assert full is not None

    assert full.content_markdown.startswith(preview)
    assert preview == preview.rstrip()
    # What follows the cut is a paragraph break, not the middle of a line.
    assert full.content_markdown[len(preview) :].startswith(("\n\n", "\n"))


async def test_a_pro_user_gets_the_whole_thing(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "pro.reader@example.com", plan=Plan.PRO)
    data = (await client.get(f"{TEMPLATES}/{PREMIUM_SLUG}")).json()["data"]

    full = await db.scalar(select(Template).where(Template.slug == PREMIUM_SLUG))
    assert full is not None
    assert data["locked"] is False
    assert data["truncated"] is False
    assert data["content_markdown"] == full.content_markdown
    assert len(data["files"]) == 4


async def test_a_free_account_is_still_gated(client: AsyncClient, db: AsyncSession) -> None:
    """Signing up is not the unlock. Paying is."""
    await _sign_in(client, db, "free.reader@example.com", plan=Plan.FREE)
    data = (await client.get(f"{TEMPLATES}/{PREMIUM_SLUG}")).json()["data"]

    assert data["locked"] is True
    assert data["files"] == []


async def test_a_free_template_is_whole_for_everyone(client: AsyncClient) -> None:
    data = (await client.get(f"{TEMPLATES}/rag-chatbot")).json()["data"]

    assert data["locked"] is False
    assert "Where this goes wrong" in data["content_markdown"]


async def test_a_premium_response_is_never_shared_cache_eligible(
    client: AsyncClient,
) -> None:
    """A shared cache holding a Pro user's body and serving it to a free one
    gives away everything the gate protects."""
    premium = await client.get(f"{TEMPLATES}/{PREMIUM_SLUG}")
    free = await client.get(f"{TEMPLATES}/rag-chatbot")

    assert "no-store" in premium.headers["cache-control"]
    assert premium.headers["cache-control"].startswith("private")
    assert free.headers["cache-control"].startswith("public")


# ── stack templates ──────────────────────────────────────────────────────────


@pytest.mark.usefixtures("seeded_catalog")
async def test_a_stack_template_carries_a_payload_the_architect_accepts(
    client: AsyncClient,
) -> None:
    """The connection that makes this a library rather than a blog: opening one
    produces a real recommendation against today's catalog."""
    data = (await client.get(f"{TEMPLATES}/rag-chatbot")).json()["data"]
    assert data["is_stack_template"] is True

    response = await client.post("/api/v1/architect/recommend", json=data["stack_input"])
    assert response.status_code == 200, response.text
    assert float(response.json()["data"]["metrics"]["score"]) > 0


async def test_a_non_stack_template_carries_no_payload(client: AsyncClient) -> None:
    data = (await client.get(f"{TEMPLATES}/ai-production-readiness")).json()["data"]
    assert data["is_stack_template"] is False
    assert data["stack_input"] == {}


# ── counters ─────────────────────────────────────────────────────────────────


async def test_a_view_increments_the_view_count_only(client: AsyncClient, db: AsyncSession) -> None:
    await client.get(f"{TEMPLATES}/rag-chatbot")
    await db.commit()

    row = await db.scalar(select(Template).where(Template.slug == "rag-chatbot"))
    assert row is not None
    await db.refresh(row)
    assert row.view_count == 1
    assert row.copy_count == 0


async def test_a_copy_increments_the_copy_count(client: AsyncClient, db: AsyncSession) -> None:
    """Separate from views on purpose: a template people open and leave says
    something different from one they take, and only the second is a reason to
    write more like it."""
    response = await client.post(f"{TEMPLATES}/rag-chatbot/copy")
    assert response.status_code == 200
    await db.commit()

    row = await db.scalar(select(Template).where(Template.slug == "rag-chatbot"))
    assert row is not None
    await db.refresh(row)
    assert row.copy_count == 1


# ── detail page ──────────────────────────────────────────────────────────────


async def test_the_detail_page_offers_somewhere_to_go_next(client: AsyncClient) -> None:
    """An empty related block is a dead end on the page most likely to be
    someone's entry point to the product."""
    data = (await client.get(f"{TEMPLATES}/rag-chatbot")).json()["data"]

    assert len(data["related"]) == 4
    assert "rag-chatbot" not in [row["slug"] for row in data["related"]]
    assert data["related_tools"]


async def test_an_unknown_slug_is_not_found(client: AsyncClient) -> None:
    assert (await client.get(f"{TEMPLATES}/no-such-template")).status_code == 404


# ── team-private templates ───────────────────────────────────────────────────


async def test_an_organization_scoped_template_is_not_in_the_public_library(
    client: AsyncClient, db: AsyncSession
) -> None:
    """M19 lands the column and the scoping; M21 lands the membership that
    could grant access. Excluding it for everyone is the safe direction — a
    team template in the public library is a much worse failure than one that
    does not appear until M21."""
    row = await db.scalar(select(Template).where(Template.slug == "cursor-rules"))
    assert row is not None
    row.organization_id = "org_private"
    await db.flush()

    assert "cursor-rules" not in await _slugs(client)
    assert (await client.get(f"{TEMPLATES}/cursor-rules")).status_code == 404


# ── seeding ──────────────────────────────────────────────────────────────────


async def test_seeding_twice_changes_nothing(db: AsyncSession) -> None:
    from app.services.seed_service import SeedReport, _seed_templates

    report = SeedReport()
    await _seed_templates(db, report, refresh=False)

    assert report.inserted["templates"] == 0
    assert report.updated["templates"] == 0
    assert report.skipped["templates"] == 30


async def test_an_edited_file_wins_on_the_next_seed(db: AsyncSession) -> None:
    """Templates break the insert-only default the rest of the seeder follows.
    A price is corrected by an editor through the review loop, so overwriting
    it would undo human work; a template has no such loop — the Markdown file
    *is* the source of truth."""
    from app.services.seed_service import SeedReport, _seed_templates

    row = await db.scalar(select(Template).where(Template.slug == "rag-chatbot"))
    assert row is not None
    row.title = "Something Someone Typed Into The Database"
    await db.flush()

    report = SeedReport()
    await _seed_templates(db, report, refresh=False)
    await db.flush()
    await db.refresh(row)

    assert report.updated["templates"] == 1
    assert row.title == "RAG Chatbot"


async def test_seeding_does_not_reset_the_counters(db: AsyncSession) -> None:
    """They are measurements, not content. Resetting them every deploy would
    destroy the only reliable input to the content roadmap."""
    from app.services.seed_service import SeedReport, _seed_templates

    row = await db.scalar(select(Template).where(Template.slug == "rag-chatbot"))
    assert row is not None
    row.view_count = 412
    row.copy_count = 37
    row.title = "Forces an update"
    await db.flush()

    await _seed_templates(db, SeedReport(), refresh=False)
    await db.flush()
    await db.refresh(row)

    assert (row.view_count, row.copy_count) == (412, 37)
