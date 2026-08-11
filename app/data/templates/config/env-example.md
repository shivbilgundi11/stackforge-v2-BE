---
title: .env.example for AI Projects
category: config
difficulty: beginner
summary: >
  Every variable an AI service actually reads, grouped by concern, with the
  placeholder convention that keeps a real key from being committed by
  autocomplete.
use_cases: [rag, chat, agents]
tags: [config, environment, secrets]
related_tools: []
---

An `.env.example` has one job beyond documentation: make it obvious at a glance
whether a value has been filled in. Every placeholder here is `change-me` or
prefixed `your-`, so a committed real key stands out in a diff instead of
blending into plausible-looking defaults.

```bash path=.env.example
# Copy to .env and replace every value. Nothing in this file is a credential.
# .env is gitignored; this file is not.

# ── Model providers ─────────────────────────────────────────────────────────
# Only the one you use. An unset key should make the feature degrade, not crash.
ANTHROPIC_API_KEY=sk-ant-change-me
OPENAI_API_KEY=sk-change-me

# Which model, as a variable rather than a literal in the code. Model names
# change, and a name hardcoded in four files changes in three of them.
CHAT_MODEL=claude-sonnet-4-5
EMBEDDING_MODEL=text-embedding-3-small
# Baked into the vector column. Changing the model means changing this AND
# re-embedding the whole corpus.
EMBEDDING_DIMENSIONS=1536

# ── Storage ─────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://postgres:change-me@localhost:5432/app
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=change-me

# ── Cache and queue ─────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0

# ── Retrieval tuning ────────────────────────────────────────────────────────
# In tokens, not characters: 1,000 characters is a very different amount of
# context in English than in code.
CHUNK_TOKENS=400
CHUNK_OVERLAP_TOKENS=60
TOP_K=5

# ── Limits ──────────────────────────────────────────────────────────────────
# A ceiling that exists in config is a ceiling you can lower during an
# incident. One that only exists in the model's judgement is not.
MAX_TOKENS_PER_REQUEST=4096
MAX_REQUESTS_PER_MINUTE=60
MONTHLY_SPEND_CAP_USD=500

# ── Observability ───────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY=pk-lf-change-me
LANGFUSE_SECRET_KEY=sk-lf-change-me
SENTRY_DSN=https://change-me@sentry.io/0
LOG_LEVEL=INFO
```

## Three conventions worth keeping

**Every model name is a variable.** They change, they get deprecated, and a
name hardcoded in four files gets updated in three.

**The embedding dimension sits next to the embedding model**, with the comment
saying they move together. Separating them is how a dimension mismatch reaches
insert time.

**Limits are configuration, not constants.** A spend cap you can lower without
a deploy is a spend cap that helps during the incident rather than after it.

## What does not belong here

Anything real. If a value in this file works, it is a leaked credential — and
`.env.example` is committed by definition.
