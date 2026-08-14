"""Small operational CLI.

uv run python -m app.cli generate-keypair
"""

from __future__ import annotations

import sys


def _generate_keypair() -> None:
    from app.services.token_service import generate_keypair

    private_pem, public_pem = generate_keypair()
    # Written as single-line escaped values so they paste straight into .env.
    print("AUTH_PRIVATE_KEY=" + private_pem.strip().replace("\n", "\\n"))
    print("AUTH_PUBLIC_KEY=" + public_pem.strip().replace("\n", "\\n"))


def _openapi() -> None:
    """Dump the schema without starting a server.

    CI generates `types/api.ts` from this rather than booting the API, so the
    drift check needs no database and no port.
    """
    import json

    from app.main import app

    print(json.dumps(app.openapi(), indent=2))


def _seed() -> None:
    """Load the catalog seed.

    Non-destructive: existing rows are left alone so an editorial correction
    survives the next deploy. `--refresh` overwrites from the seed files, for
    when the seed file *is* the correction.
    """
    import asyncio

    refresh = "--refresh" in sys.argv

    async def run() -> None:
        from app.core.database import SessionLocal
        from app.services.seed_service import seed_all

        async with SessionLocal() as session:
            report = await seed_all(session, refresh=refresh)
            await session.commit()

        for table in sorted(set(report.inserted) | set(report.updated)):
            print(
                f"{table:24} +{report.inserted.get(table, 0):<6} "
                f"~{report.updated.get(table, 0):<6} "
                f"={report.skipped.get(table, 0)}"
            )
        print(f"\ninserted {report.total_inserted}, updated {report.total_updated}")
        if report.price_changes:
            print(f"{report.price_changes} price change(s) recorded in pricing_history")

        if report.unmanaged:
            print(f"\n{len(report.unmanaged)} row(s) the seed no longer describes:")
            for row in report.unmanaged:
                print(f"  {row}")
            print("Usually a renamed id. Nothing was deleted — decide and edit the seed.")

    asyncio.run(run())


def _set_plan() -> None:
    """set-plan <email> <plan> — grant a plan without a checkout.

    The operator path for manual grants (an Enterprise deal, a comp, a
    support fix) and what the team E2E suite uses to reach the Team tier
    without Razorpay keys. Sets `plan_source` to personal, exactly as a paid
    personal subscription would.
    """
    import asyncio

    if len(sys.argv) < 4:
        print("usage: python -m app.cli set-plan <email> <free|pro|team|enterprise>")
        raise SystemExit(2)
    email, plan_value = sys.argv[2], sys.argv[3]

    async def run() -> None:
        from sqlalchemy import select

        from app.core.database import SessionLocal
        from app.models.user import Plan, PlanSource, User

        async with SessionLocal() as session:
            user = (
                await session.execute(select(User).where(User.email == email))
            ).scalar_one_or_none()
            if user is None:
                print(f"no user with email {email}")
                raise SystemExit(1)
            user.plan = Plan(plan_value)
            user.plan_source = PlanSource.PERSONAL
            await session.commit()
            print(f"{email} -> {plan_value}")

    asyncio.run(run())


def _invite_link() -> None:
    """invite-link <email> — print a fresh accept link for an open invitation.

    The operator answer to "the invite never arrived": rotates the token
    (the emailed link dies, exactly like a resend) and prints the new URL.
    Also how the E2E suite reads an invite link without a mailbox.
    """
    import asyncio
    from datetime import timedelta

    if len(sys.argv) < 3:
        print("usage: python -m app.cli invite-link <email>")
        raise SystemExit(2)
    email = sys.argv[2]

    async def run() -> None:
        from sqlalchemy import select

        from app.core.config import settings
        from app.core.database import SessionLocal, utcnow
        from app.models.organization import Invitation
        from app.services import token_service

        async with SessionLocal() as session:
            invitation = (
                await session.execute(
                    select(Invitation)
                    .where(
                        Invitation.email == email,
                        Invitation.accepted_at.is_(None),
                        Invitation.revoked_at.is_(None),
                    )
                    .order_by(Invitation.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if invitation is None:
                print(f"no open invitation for {email}")
                raise SystemExit(1)
            token = token_service.generate_secret()
            invitation.token_hash = token_service.hash_secret(token)
            invitation.expires_at = utcnow() + timedelta(days=settings.invite_ttl_days)
            await session.commit()
            print(f"{settings.web_base_url}/invite?token={token}")

    asyncio.run(run())


def _purge_runs() -> None:
    """Delete unsaved runs past the retention window.

    The scheduled job behind M17's save model: everything is logged, and what
    the user did not choose to keep expires. `--dry-run` reports the count
    without deleting, because the first time anyone runs a destructive job
    against production they should be able to see what it would do.
    """
    import asyncio

    dry_run = "--dry-run" in sys.argv

    async def run() -> None:
        from datetime import timedelta

        from sqlalchemy import func, select

        from app.core.database import SessionLocal, utcnow
        from app.models.tool_run import ToolRun
        from app.services.run_service import RETENTION_DAYS, purge_expired

        async with SessionLocal() as session:
            cutoff = utcnow() - timedelta(days=RETENTION_DAYS)
            doomed = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ToolRun)
                    .where(ToolRun.saved.is_(False), ToolRun.created_at < cutoff)
                )
                or 0
            )
            kept = int(
                await session.scalar(
                    select(func.count()).select_from(ToolRun).where(ToolRun.saved.is_(True))
                )
                or 0
            )

            if dry_run:
                print(f"would delete {doomed} unsaved run(s) older than {RETENTION_DAYS} days")
                print(f"{kept} saved run(s) are exempt")
                return

            removed = await purge_expired(session)
            await session.commit()
            print(f"deleted {removed} unsaved run(s) older than {RETENTION_DAYS} days")
            print(f"{kept} saved run(s) untouched")

    asyncio.run(run())


def _razorpay_sync() -> None:
    """Create the plans this build sells.

    In a script rather than the dashboard, so environments are reproducible and
    a plan id is never a thing someone remembers copying. Idempotent: plans are
    matched on period, interval, and amount, so running it twice creates
    nothing and prints the same ids.

    It never *edits* a plan. Razorpay plans are immutable once a subscription
    is on them — changing what something costs means creating a new plan and
    pointing config at it, which is exactly the behaviour you want when a
    subscriber is already on the old one.

    Prints the `.env` lines to paste. Deliberately not written to a file: these
    are environment configuration, and a script that edits `.env` behind
    someone's back is a script that eventually edits the wrong one.
    """
    import razorpay

    from app.core.config import settings
    from app.data import plans as plan_data

    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        print("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set.", file=sys.stderr)
        raise SystemExit(2)

    live = not settings.razorpay_key_id.startswith("rzp_test_")
    if live and "--live" not in sys.argv:
        print("That is a live key. Re-run with --live if you mean it.", file=sys.stderr)
        raise SystemExit(2)

    client = razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))

    # One page is plenty — this account has a handful of plans, and paging
    # would be inventing a problem to solve.
    existing = client.plan.all({"count": 100}).get("items", [])
    lines: list[str] = []

    for spec in plan_data.PLANS:
        if not spec.checkout or spec.monthly_minor is None:
            continue

        for period, amount, suffix in (
            ("monthly", spec.monthly_minor, "MONTHLY"),
            ("yearly", spec.annual_minor, "ANNUAL"),
        ):
            if amount is None:
                continue

            match = next(
                (
                    plan
                    for plan in existing
                    if plan.get("period") == period
                    and plan.get("interval") == 1
                    and plan.get("item", {}).get("amount") == amount
                    and plan.get("item", {}).get("currency") == plan_data.CURRENCY.upper()
                    and plan.get("notes", {}).get("stackforge_plan") == spec.plan.value
                ),
                None,
            )
            if match is not None:
                plan_id = str(match["id"])
                print(f"{spec.label:12} {period:7} plan exists  {plan_id}")
            else:
                created = client.plan.create(
                    {
                        "period": period,
                        "interval": 1,
                        "item": {
                            "name": f"StackForge {spec.label} ({period})",
                            "description": spec.tagline,
                            "amount": amount,
                            "currency": plan_data.CURRENCY.upper(),
                        },
                        "notes": {"stackforge_plan": spec.plan.value},
                    }
                )
                plan_id = str(created["id"])
                print(f"{spec.label:12} {period:7} plan created {plan_id}")

            lines.append(f"RAZORPAY_PLAN_{spec.plan.value.upper()}_{suffix}={plan_id}")

    print("\nPaste into .env:\n")
    for line in lines:
        print(line)


COMMANDS = {
    "generate-keypair": _generate_keypair,
    "openapi": _openapi,
    "seed": _seed,
    "set-plan": _set_plan,
    "invite-link": _invite_link,
    "purge-runs": _purge_runs,
    "razorpay-sync": _razorpay_sync,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python -m app.cli <command>", file=sys.stderr)
        print("commands: " + ", ".join(COMMANDS), file=sys.stderr)
        return 2
    COMMANDS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
