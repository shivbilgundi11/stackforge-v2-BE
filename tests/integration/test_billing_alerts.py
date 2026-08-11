"""The two retention emails, and who does not get them (M20).

These are the product's structural retention mechanism (`PRD.md` §24): the data
moves on its own, which gives a user a reason to come back that does not depend
on a habit. That only works if the mail is *right* — an alert about something
the reader did not save is noise, and a channel that sends noise is a channel
people filter.

So the assertions here are as much about who is *not* emailed as who is.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import new_id, utcnow
from app.models.catalog import ModelPricing, PricedEntity, PricingHistory, Tool, ToolStatus
from app.models.stack import Stack
from app.models.tool_run import ToolRun
from app.models.user import Plan, User
from app.workers import billing as billing_jobs

pytestmark = pytest.mark.usefixtures("seeded_catalog")


async def _user(db: AsyncSession, email: str, plan: Plan) -> User:
    user = User(id=new_id("usr"), email=email, name=email.split("@")[0], plan=plan)
    db.add(user)
    await db.flush()
    return user


async def _saved_run(db: AsyncSession, user: User, model_id: str) -> ToolRun:
    run = ToolRun(
        id=new_id("run"),
        tool_slug="llm-pricing",
        workflow="cost",
        user_id=user.id,
        input={"model_id": model_id, "input_tokens": 1000, "output_tokens": 500},
        output={},
        duration_ms=5,
        saved=True,
        created_at=utcnow(),
    )
    db.add(run)
    await db.flush()
    return run


async def _drift(db: AsyncSession, model: ModelPricing, pct: str) -> PricingHistory:
    row = PricingHistory(
        id=new_id("ph"),
        entity_type=PricedEntity.MODEL,
        entity_id=model.id,
        field="input_cost_per_1k",
        old_value=Decimal("0.000150"),
        new_value=Decimal("0.000225"),
        pct_change=Decimal(pct),
        applied=False,
        source_id=model.source_id,
        detected_at=utcnow() - timedelta(hours=1),
    )
    db.add(row)
    await db.flush()
    return row


async def _a_model(db: AsyncSession) -> ModelPricing:
    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    return model


# ── Price change ────────────────────────────────────────────────────────────


async def test_a_pro_user_with_an_affected_saved_run_is_emailed(
    db: AsyncSession, outbox: Any
) -> None:
    model = await _a_model(db)
    user = await _user(db, "pro-alerts@example.com", Plan.PRO)
    await _saved_run(db, user, model.model_id)
    await _drift(db, model, "50")

    assert await billing_jobs.send_price_change_alerts(db) == 1

    mail = outbox.outbox[-1]
    assert mail.to == user.email
    assert "changed" in mail.subject
    # The figures are in the body: an alert that only says "come and look"
    # converts a click from the people who least needed it.
    assert "0.000150" in mail.text
    assert "0.000225" in mail.text
    assert "up 50" in mail.text


async def test_a_free_user_is_not_emailed(db: AsyncSession, outbox: Any) -> None:
    """Alerts are a Pro feature. Sending them to Free would give away the thing
    being sold and train the reader to ignore the channel."""
    model = await _a_model(db)
    user = await _user(db, "free-alerts@example.com", Plan.FREE)
    await _saved_run(db, user, model.model_id)
    await _drift(db, model, "50")

    assert await billing_jobs.send_price_change_alerts(db) == 0
    assert outbox.outbox == []


async def test_a_small_move_sends_nothing(db: AsyncSession, outbox: Any) -> None:
    """FR-18 is ten percent. A one-percent move is not news, and a channel that
    reports it stops being read."""
    model = await _a_model(db)
    user = await _user(db, "small@example.com", Plan.PRO)
    await _saved_run(db, user, model.model_id)
    await _drift(db, model, "2.5")

    assert await billing_jobs.send_price_change_alerts(db) == 0
    assert outbox.outbox == []


async def test_an_unsaved_run_is_not_an_affected_estimate(db: AsyncSession, outbox: Any) -> None:
    """An unsaved run is deleted after 30 days and was never something the user
    said they cared about."""
    model = await _a_model(db)
    user = await _user(db, "unsaved@example.com", Plan.PRO)
    run = await _saved_run(db, user, model.model_id)
    run.saved = False
    await db.flush()
    await _drift(db, model, "40")

    assert await billing_jobs.send_price_change_alerts(db) == 0


async def test_a_user_whose_saved_work_is_unaffected_is_not_emailed(
    db: AsyncSession, outbox: Any
) -> None:
    model = await _a_model(db)
    other = await _user(db, "elsewhere@example.com", Plan.PRO)
    await _saved_run(db, other, "some-other-model")
    await _drift(db, model, "80")

    assert await billing_jobs.send_price_change_alerts(db) == 0


async def test_an_old_change_is_outside_the_window(db: AsyncSession, outbox: Any) -> None:
    """Otherwise every run would re-send the whole history."""
    model = await _a_model(db)
    user = await _user(db, "stale-window@example.com", Plan.PRO)
    await _saved_run(db, user, model.model_id)
    change = await _drift(db, model, "60")
    change.detected_at = utcnow() - timedelta(days=14)
    await db.flush()

    assert await billing_jobs.send_price_change_alerts(db) == 0


# ── Deprecation ─────────────────────────────────────────────────────────────


async def test_a_deprecated_tool_in_a_saved_stack_is_emailed_once_per_tool(
    db: AsyncSession, outbox: Any
) -> None:
    """A user with the same dead tool in four stacks needs telling about the
    tool, not about the four stacks."""
    tool = (await db.execute(select(Tool).limit(1))).scalars().first()
    assert tool is not None
    tool.status = ToolStatus.DEPRECATED
    tool.status_reason = "Superseded and unmaintained."
    await db.flush()

    user = await _user(db, "deprecated@example.com", Plan.PRO)
    for index in range(3):
        db.add(
            Stack(
                id=new_id("stk"),
                user_id=user.id,
                name=f"Stack {index}",
                requirements={},
                component_slugs=[tool.slug],
            )
        )
    await db.flush()

    from app.services import catalog_service

    await catalog_service.invalidate()

    assert await billing_jobs.send_deprecation_alerts(db) == 1

    mail = outbox.outbox[-1]
    assert mail.to == user.email
    assert mail.text.count(tool.name) == 1
    assert "deprecated" in mail.text


async def test_a_free_user_gets_no_deprecation_email(db: AsyncSession, outbox: Any) -> None:
    tool = (await db.execute(select(Tool).limit(1))).scalars().first()
    assert tool is not None
    tool.status = ToolStatus.DEPRECATED
    await db.flush()

    user = await _user(db, "free-deprecated@example.com", Plan.FREE)
    db.add(
        Stack(
            id=new_id("stk"),
            user_id=user.id,
            name="Free stack",
            requirements={},
            component_slugs=[tool.slug],
        )
    )
    await db.flush()

    from app.services import catalog_service

    await catalog_service.invalidate()

    assert await billing_jobs.send_deprecation_alerts(db) == 0


async def test_a_healthy_stack_sends_nothing(db: AsyncSession, outbox: Any) -> None:
    user = await _user(db, "healthy@example.com", Plan.PRO)
    tool = (
        (await db.execute(select(Tool).where(Tool.status == ToolStatus.RECOMMENDED).limit(1)))
        .scalars()
        .first()
    )
    assert tool is not None
    db.add(
        Stack(
            id=new_id("stk"),
            user_id=user.id,
            name="Fine",
            requirements={},
            component_slugs=[tool.slug],
        )
    )
    await db.flush()

    assert await billing_jobs.send_deprecation_alerts(db) == 0
