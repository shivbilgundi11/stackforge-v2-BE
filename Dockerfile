# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# StackForge API — one image, three roles
#
#   api      uvicorn app.main:app
#   worker   arq app.workers.queue.WorkerSettings
#   migrate  alembic upgrade head  (one-shot, on every deploy)
#
# They share an image because they share a code path: the worker imports the
# same services the API does, and an export built in the request and one built
# in the queue must render identically or the feature has two behaviours.
#
# Chromium is installed here rather than in a separate PDF image because the
# sync export path (anything under EXPORT_ASYNC_THRESHOLD_BYTES) renders
# inside the API request. Set PDF_BACKEND=reportlab to run without it; the
# layer is still present, just unused.
# ─────────────────────────────────────────────────────────────────────────────

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependency layer first: it is invalidated only by a lockfile change, so an
# ordinary code deploy reuses it.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Playwright + Chromium. `--with-deps` pulls the ~90 system libraries a
# headless browser needs; without it the browser installs and then refuses to
# launch, which surfaces as a PDF backend that silently falls back.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install playwright \
 && playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/*

COPY . .

# Non-root. Chromium's cache was written to /root by the install above, so it
# moves with the ownership change.
RUN groupadd --system --gid 1001 app \
 && useradd  --system --uid 1001 --gid app --home /home/app --create-home app \
 && mv /root/.cache/ms-playwright /home/app/.cache-ms-playwright 2>/dev/null || true \
 && mkdir -p /home/app/.cache \
 && mv /home/app/.cache-ms-playwright /home/app/.cache/ms-playwright 2>/dev/null || true \
 && chown -R app:app /app /home/app

USER app
ENV HOME=/home/app

EXPOSE 8000

# Overridden per service in compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
