---
title: FastAPI RAG Service
category: code-starter
difficulty: intermediate
summary: >
  A retrieval endpoint that runs: pgvector storage, an ingest path, streaming
  answers, and the two things starters usually omit — a real chunker and a
  citation in the response.
use_cases: [rag, chat]
tags: [fastapi, pgvector, python, streaming]
related_tools: [chunking-strategy, embedding-cost, vectordb-estimate]
premium: true
---

A working retrieval service in four files. It is deliberately not a framework:
there is no chain abstraction and no agent loop, because the thing most RAG
projects need first is a clear view of what text went into the prompt.

Two things here that tutorials usually skip.

**The chunker respects structure.** Splitting on a character count cuts
sentences in half and strands headings from the paragraphs they introduce.
This one splits on paragraph boundaries and only falls back to a hard cut when
a single paragraph exceeds the window.

**The response carries citations.** Every answer returns the chunk ids it was
built from. Without that, "the model made this up" and "the retrieval returned
the wrong document" look identical from the outside, and they need completely
different fixes.

## Running it

```bash
uv sync
docker compose up -d postgres
export OPENAI_API_KEY=sk-...
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Then `POST /ingest` with a document and `POST /ask` with a question.

```python path=app/config.py
"""Settings, read once at import."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag"
    openai_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    # 1536 for text-embedding-3-small. Changing the model means changing this
    # AND re-embedding everything — the dimension is baked into the column.
    embedding_dimensions: int = 1536
    chat_model: str = "gpt-4o-mini"

    # Tokens, not characters. A 1,000-character chunk is a very different
    # amount of context in English than in code.
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60
    top_k: int = 5

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python path=app/chunking.py
"""Structure-aware chunking.

Splitting on a fixed character count is the default in every tutorial and it
is the single largest avoidable loss of retrieval quality: it cuts sentences
in half and separates a heading from the paragraph it introduces, so the chunk
that gets retrieved is missing the context that made it findable.

This splits on paragraph boundaries, packs paragraphs up to the token budget,
and only hard-cuts when one paragraph is itself too large.
"""

from __future__ import annotations

import re

import tiktoken

PARAGRAPH = re.compile(r"\n\s*\n")


def _encoder(model: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(model)


def chunk(text: str, *, max_tokens: int = 400, overlap_tokens: int = 60) -> list[str]:
    encoder = _encoder()
    paragraphs = [part.strip() for part in PARAGRAPH.split(text) if part.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        tokens = len(encoder.encode(paragraph))

        if tokens > max_tokens:
            # One paragraph larger than the window. Flush what we have, then
            # hard-split this one — there is no structural boundary left to use.
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_hard_split(paragraph, encoder, max_tokens, overlap_tokens))
            continue

        if current_tokens + tokens > max_tokens:
            chunks.append("\n\n".join(current))
            # Carry the last paragraph forward so a fact spanning a boundary is
            # retrievable from either side.
            current = [current[-1]] if current and overlap_tokens else []
            current_tokens = len(encoder.encode(current[0])) if current else 0

        current.append(paragraph)
        current_tokens += tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _hard_split(text: str, encoder: tiktoken.Encoding, max_tokens: int, overlap: int) -> list[str]:
    tokens = encoder.encode(text)
    step = max(1, max_tokens - overlap)
    return [
        encoder.decode(tokens[start : start + max_tokens]) for start in range(0, len(tokens), step)
    ]
```

```python path=app/store.py
"""pgvector storage and retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg
from openai import AsyncOpenAI

from app.config import get_settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document    text NOT NULL,
    content     text NOT NULL,
    embedding   vector(%(dimensions)s) NOT NULL
);

-- Without this index pgvector does a sequential scan. It stays fast enough on
-- a few thousand rows to hide the problem until the corpus is real.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


@dataclass(frozen=True)
class Retrieved:
    id: int
    document: str
    content: str
    distance: float


class Store:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._client = AsyncOpenAI(api_key=get_settings().openai_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        # Batched: one request per chunk turns a 500-chunk ingest into 500
        # round trips and 500 chances to hit a rate limit.
        response = await self._client.embeddings.create(model=settings.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    async def add(self, document: str, chunks: list[str]) -> int:
        vectors = await self.embed(chunks)
        async with self._pool.acquire() as connection:
            await connection.executemany(
                "INSERT INTO chunks (document, content, embedding) VALUES ($1, $2, $3)",
                [
                    (document, content, str(vector))
                    for content, vector in zip(chunks, vectors, strict=True)
                ],
            )
        return len(chunks)

    async def search(self, question: str, *, top_k: int = 5) -> list[Retrieved]:
        vector = (await self.embed([question]))[0]
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT id, document, content, embedding <=> $1 AS distance
                FROM chunks
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                str(vector),
                top_k,
            )
        return [Retrieved(**dict(row)) for row in rows]
```

```python path=app/main.py
"""The two endpoints: ingest and ask."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import asyncpg
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.chunking import chunk
from app.config import get_settings
from app.store import SCHEMA, Store

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pool = await asyncpg.create_pool(settings.database_url.replace("+asyncpg", ""))
    async with pool.acquire() as connection:
        await connection.execute(SCHEMA % {"dimensions": settings.embedding_dimensions})
    app.state.store = Store(pool)
    yield
    await pool.close()


app = FastAPI(title="RAG service", lifespan=lifespan)


class IngestIn(BaseModel):
    document: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.post("/ingest")
async def ingest(payload: IngestIn) -> dict[str, int]:
    chunks = chunk(
        payload.text,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    stored = await app.state.store.add(payload.document, chunks)
    return {"chunks": stored}


@app.post("/ask")
async def ask(payload: AskIn) -> StreamingResponse:
    retrieved = await app.state.store.search(payload.question, top_k=settings.top_k)

    context = "\n\n---\n\n".join(
        f"[{item.id}] {item.document}\n{item.content}" for item in retrieved
    )
    # The citation instruction is in the prompt AND the ids are returned in the
    # header. Asking the model to cite is not the same as knowing what it saw:
    # without the header, "it hallucinated" and "retrieval returned the wrong
    # document" look identical from outside and need opposite fixes.
    prompt = (
        "Answer using only the context below. Cite the bracketed ids you used. "
        "If the context does not contain the answer, say so plainly.\n\n"
        f"{context}\n\nQuestion: {payload.question}"
    )

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def stream() -> AsyncIterator[str]:
        completion = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for part in completion:
            token = part.choices[0].delta.content
            if token:
                yield token

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Retrieved-Chunks": ",".join(str(item.id) for item in retrieved)},
    )
```

## What is deliberately missing

No reranking, no hybrid search, no query rewriting. All three help, and all
three are tuning you should do *after* you have an evaluation set and a
baseline — adding them first means you cannot tell which one earned the
improvement.
