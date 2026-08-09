# StackForge API

FastAPI backend for the StackForge AI engineering workbench.

## Requirements

- Python 3.13 (managed by `uv`)
- PostgreSQL 16+ with the `citext` extension
- Redis 7 *(optional today — nothing in the auth layer needs it yet)*

## Setup

```bash
uv sync
cp .env.example .env

# Generate the Ed25519 signing keypair and paste both lines into .env
uv run python -m app.cli generate-keypair

# Create the databases (or use `docker compose up -d`)
createdb stackforge_v2
createdb stackforge_test
psql -d stackforge_v2 -c 'CREATE EXTENSION IF NOT EXISTS citext'
psql -d stackforge_test -c 'CREATE EXTENSION IF NOT EXISTS citext'

uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

`docker compose up -d` brings up Postgres, Redis, and Mailpit if you have
Docker. Mailpit catches every outbound email at <http://localhost:8025>.

Without Docker, set `EMAIL_PROVIDER=console` and verification and reset links
are printed to the API log.

## Commands

| | |
| --- | --- |
| `uv run uvicorn app.main:app --reload` | Run the API |
| `uv run pytest` | Test suite |
| `uv run pytest --cov=app` | With coverage |
| `uv run ruff check app tests` | Lint |
| `uv run ruff format app tests` | Format |
| `uv run mypy app` | Type check |
| `uv run alembic revision --autogenerate -m "…"` | New migration |
| `uv run alembic upgrade head` | Apply migrations |
| `uv run python -m app.cli openapi` | Dump the OpenAPI schema |

## Layout

```
app/
  core/          config, database, redis, errors, logging, middleware
  api/v1/        routers — HTTP only, no business logic
  models/        SQLAlchemy declarative
  schemas/       Pydantic request/response
  services/      all business logic; formulas live here and nowhere else
  integrations/  anything that talks to a third party
alembic/         migrations
tests/           unit (no database) + integration (real Postgres)
```

## Conventions

- **Routers contain no formulas.** Parse, authorise, call one service, return.
- **Money is `Decimal` / `NUMERIC(14,6)`**, never float.
- **Every response uses the envelope** and declares `response_model=Envelope[T]`
  — the frontend's types are generated from that schema.
- **Every timestamp is timezone-aware.** Use `utcnow()`.
- **Tests assert computed values**, not status codes.
- Migrations are expand-then-contract, and the downgrade must drop enum types
  or the round trip fails.
