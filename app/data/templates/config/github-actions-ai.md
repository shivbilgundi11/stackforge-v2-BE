---
title: GitHub Actions for AI Projects
category: config
difficulty: intermediate
summary: >
  CI that catches the failures specific to AI codebases: a suite that silently
  calls a live model, a lockfile that drifts, and a prompt change nobody
  reviewed.
use_cases: [coding, automation]
tags: [ci, github-actions, testing, python]
related_tools: []
---

Standard Python CI plus three checks that only matter when there is a model
involved.

```yaml path=.github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready
          --health-interval 5s
          --health-timeout 3s
          --health-retries 10
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 5s
          --health-retries 10

    env:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/test
      REDIS_URL: redis://localhost:6379/0
      # Deliberately empty. The suite must pass with no key, and a job that
      # provides one cannot prove that. See the guard step below.
      OPENAI_API_KEY: ""
      ANTHROPIC_API_KEY: ""

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install
        # --frozen fails if the lockfile does not match pyproject.toml. Without
        # it, CI resolves fresh dependencies and passes against versions nobody
        # has locally.
        run: uv sync --frozen

      - name: Lint
        run: uv run ruff check app tests

      - name: Format
        run: uv run ruff format --check app tests

      - name: Types
        run: uv run mypy app

      - name: Migrations are in step with the models
        run: |
          uv run alembic upgrade head
          uv run alembic check

      - name: Test
        run: uv run pytest -q

      - name: No live model calls in the suite
        # A test that calls a real model is billable, non-deterministic, and
        # fails in a fork with no secrets. Grepping for the client constructors
        # outside the fixtures catches the one someone adds in a hurry.
        run: |
          if grep -rn --include=*.py -E "(AsyncOpenAI|Anthropic)\(" tests/ \
             | grep -v "tests/conftest.py" | grep -v "monkeypatch"; then
            echo "::error::A test constructs a live model client. Mock it."
            exit 1
          fi

  prompt-review:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Flag prompt changes for review
        # A prompt is behaviour, and a prompt diff buried in a 40-file PR gets
        # scrolled past. This makes it a visible line in the checks.
        run: |
          if git diff --name-only origin/${{ github.base_ref }}...HEAD \
             | grep -E "prompts?/|_prompts\.py$"; then
            echo "::warning::This PR changes prompts. Review the diff deliberately."
          fi
```

## The three AI-specific checks

**The suite must pass with no API key.** A developer with a key in their `.env`
otherwise runs a different suite from CI, makes live billable calls on every
`pytest`, and gets non-deterministic failures. Clearing the keys in the job
environment makes "passes with the key unset" true by construction.

**`uv sync --frozen`.** AI dependency trees move fast, and a CI that resolves
fresh versions is a CI that tests something nobody has installed.

**`alembic check`.** Catches a model changed without a migration — the failure
that passes every test locally and breaks the first deploy.
