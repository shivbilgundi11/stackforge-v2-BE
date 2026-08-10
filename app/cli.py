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


COMMANDS = {
    "generate-keypair": _generate_keypair,
    "openapi": _openapi,
    "seed": _seed,
    "purge-runs": _purge_runs,
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
