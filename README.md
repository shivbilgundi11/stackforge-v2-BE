# StackForge API

FastAPI backend for the StackForge AI engineering workbench.

## Requirements

- Python 3.13 (managed by `uv`)
- PostgreSQL 16+ with the `citext` extension
- Redis 7 *(quota counters, the catalog cache, and the job queue — every one of
  them degrades rather than fails without it, so the API still boots and serves)*

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
| `uv run arq app.workers.queue.WorkerSettings` | Run the background worker |
| `uv run pytest` | Test suite |
| `uv run pytest --cov=app` | With coverage |
| `uv run ruff check app tests` | Lint |
| `uv run ruff format app tests` | Format |
| `uv run mypy app` | Type check |
| `uv run alembic revision --autogenerate -m "…"` | New migration |
| `uv run alembic upgrade head` | Apply migrations |
| `uv run python -m app.cli openapi` | Dump the OpenAPI schema |
| `uv run python -m app.cli seed` | Load the catalog, templates, and plan quotas |
| `uv run python -m app.cli razorpay-sync` | Create the Razorpay plans |

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

## Templates

The library lives in `app/data/templates/<category>/<slug>.md` — Markdown with
YAML frontmatter, seeded by `uv run python -m app.cli seed`. Adding one is a
file plus a seed run; nothing is registered in code.

Multi-file code starters carry their files as fenced blocks tagged with a path:

    ```python path=app/main.py

The seeder overwrites a changed file without `--refresh`, unlike every other
seed in this repo (D-43): the file is the source of truth, and there is no
editorial review loop for a template to undo. View and copy counts are never
reset.

## Billing and quotas

**Every limit is a row in `plan_quotas`, not a constant.** Changing the free
tier from 25 runs to 10 is one `UPDATE` and takes effect within a minute — the
limits are cached in-process for 60 seconds and nothing else needs restarting.
The pricing page reads the same table, so a marketing number and an enforced
number cannot drift apart.

`FeatureService` is the only place a plan question is answered:

```python
feature_service.can(identity, Feature.EXPORT_PDF)              # Allow | Deny
feature_service.check(db, identity, Metric.TOOL_RUNS_PER_DAY)  # QuotaState
feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)  # or raises 402
```

Routes use the `require_feature(...)` and `consume_quota(...)` dependencies. No
route contains a plan comparison, and no service keeps its own limit table.

Razorpay is optional. Without `RAZORPAY_ENABLED` and the keys the module
imports, checkout returns a 402 that says so, and the pricing page hides its buy
buttons — which is the state local development and CI run in. To exercise the
real path:

```bash
uv run python -m app.cli razorpay-sync   # creates the plans, prints the ids
cloudflared tunnel --url http://localhost:8000   # any tunnel will do
```

Razorpay ships no CLI that forwards webhooks, so the tunnel URL has to be
registered once under Dashboard → Settings → Webhooks, pointing at
`/api/v1/billing/webhook` with the secret you already put in
`RAZORPAY_WEBHOOK_SECRET` (D-50). Until that exists, checkout completes and
nothing upgrades.

Webhook deliveries are recorded in `billing_events` **before** they are
processed, and `processed_at` — not the row's existence — is what marks one
done (D-45). A handler that raises leaves an unprocessed row with its error, an
hourly job retries it, and the endpoint still answers 200 so Razorpay does not
disable it.

## Background work

The worker (`arq`) owns the export jobs and the billing clock. Nothing here is
required for the API to serve traffic — an export that cannot be queued is
built inside the request instead — so local development without a worker is a
supported state, just a slower and untidier one.

| Job | Schedule | What it does |
| --- | --- | --- |
| `build_export` | on demand | Renders a bundle predicted to be large |
| `purge_expired_exports` | 03:17 daily | Reclaims storage from expired exports |
| `retry_billing_events` | hourly | Re-runs webhook deliveries whose handler failed |
| `expire_trials` | 02:11 daily | Drops an expired no-card trial to Free |
| `close_dunning` | 02:29 daily | Downgrades a payment that never recovered |
| `reconcile_usage` | 23:47 daily | Compares the Redis counters against `usage_records` |
| `price_change_alerts` | 08:13 daily | Emails Pro+ users whose saved estimates moved >10 % |
| `deprecation_alerts` | Mondays 08:41 | Emails Pro+ users whose saved stacks hold a buried tool |

Reconciliation **reports** divergence and never corrects it. A drift means the
metering is wrong, and silently fixing the number removes the only signal that
says so.

## PDF export

Two backends behind one interface (D-41). `PDF_BACKEND=auto`, the default, uses
headless Chromium when Playwright is importable and falls back to ReportLab
otherwise, logging which it chose.

Chromium is the production path and produces the client-ready output the
feature is sold on. It is **not** installed by `uv sync`, because it costs a
~400 MB image the API does not need:

```bash
uv pip install playwright && uv run playwright install chromium
```

Without it, exports still work — they just look like a generated report rather
than a designed one. Pin `PDF_BACKEND=chromium` in any environment where the
downgrade must be an error rather than a log line.

## Diagrams

Every generated diagram carries two things beyond its boxes and arrows, and
both are in the Mermaid source rather than bolted on by one renderer:

- **Colour by role**, as a `classDef` block. Ordinary Mermaid, so the `.mmd`
  someone downloads renders in colour on GitHub or in a VS Code preview with no
  help from us. The colour is the *role*, so the model is violet and the stores
  are blue whichever tool filled the slot.
- **Brand marks**, as `%% brand:<node>:<icon>:<hex>` comments. Comments are
  ignored by every renderer, so the artefact stays portable; a renderer that
  understands them draws the logo. The logo is not a node image because an
  `img` shape needs a data URI in the source, and five kilobytes of base64 in
  the middle of a file people read and edit is not a trade worth making.

The catalog-slug-to-icon map is `app/data/brands.py`. It covers 47 of 88 tools
— the set these are drawn from delists a brand on trademark request, and has
done so for most of the large vendors — and an unmatched tool gets a monogram
in the role's colour rather than a gap.

`app/data/brand_marks.json` holds the paths, generated from `simple-icons` and
written to both repositories at once:

```bash
cd ../frontend && npm install --no-save simple-icons && npm run brand:marks
```

The Chromium backend renders diagrams into the PDF; ReportLab prints the source
it was drawn from, because it has no browser. That needs the Mermaid bundle
vendored at `app/static/mermaid.min.js` — committed rather than fetched,
because an export must not depend on a CDN being reachable. `MERMAID_VERSION`
records the build; the web app resolves its own copy from `package.json`, and
the two should be bumped together or the same diagram renders two ways.

## Conventions

- **Routers contain no formulas.** Parse, authorise, call one service, return.
- **Money is `Decimal` / `NUMERIC(14,6)`**, never float.
- **Every response uses the envelope** and declares `response_model=Envelope[T]`
  — the frontend's types are generated from that schema.
- **Every timestamp is timezone-aware.** Use `utcnow()`.
- **Tests assert computed values**, not status codes.
- Migrations are expand-then-contract, and the downgrade must drop enum types
  or the round trip fails.
- **No plan comparison outside `FeatureService`.** A `PLAN_RANK` lookup in a
  service is how "what does Free get" became a question with five answers.
- **Unlimited is `None`, never a large number** (D-47). A sentinel renders as a
  real limit and invites arithmetic that treats it as one.
