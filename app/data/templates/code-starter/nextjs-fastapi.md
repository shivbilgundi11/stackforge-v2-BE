---
title: Next.js + FastAPI Starter
category: code-starter
difficulty: intermediate
summary: >
  A typed frontend against a Python AI backend, with the two things that
  actually bite: streaming through to the browser, and types generated from the
  API rather than hand-written.
use_cases: [chat, rag]
tags: [nextjs, fastapi, typescript, streaming]
related_tools: [llm-pricing, token-calculator]
---

The split most AI products end up with: Python where the model libraries are,
TypeScript where the interface is. Two decisions here are worth copying even if
you throw the rest away.

**Types are generated from the OpenAPI schema, never hand-written.** A renamed
field becomes a compile error in the component that reads it, rather than
`undefined` on a page at runtime. One npm script, and it pays for itself the
first time the backend changes.

**Streaming goes all the way through.** A backend that streams into a frontend
that awaits the whole response is a backend that does not stream. The route
handler below pipes the body straight through rather than buffering it.

## Running it

```bash
# backend
uv sync && uv run uvicorn app.main:app --reload
# frontend
npm install && npm run api:types && npm run dev
```

```python path=backend/app/main.py
"""A streaming chat endpoint with a typed contract."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="AI backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

client = AsyncOpenAI()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str = Field(min_length=1, max_length=32_000)


class ChatIn(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=50)
    model: str = "gpt-4o-mini"


@app.post("/chat")
async def chat(payload: ChatIn) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        completion = await client.chat.completions.create(
            model=payload.model,
            messages=[message.model_dump() for message in payload.messages],
            stream=True,
        )
        async for part in completion:
            token = part.choices[0].delta.content
            if token:
                yield token

    # `text/plain` rather than SSE: the browser can read this with a plain
    # ReadableStream, with no event parsing and no client library.
    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
```

```typescript path=frontend/app/api/chat/route.ts
// Proxied through a route handler rather than called from the browser, so the
// backend URL and any credential stay server-side. The body is piped straight
// through — awaiting it here would buffer the whole response and undo the
// streaming the backend just did.

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: Request): Promise<Response> {
  const upstream = await fetch(`${BACKEND}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
  });

  if (!upstream.ok || upstream.body === null) {
    return new Response("Upstream failed", { status: 502 });
  }

  return new Response(upstream.body, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
```

```tsx path=frontend/components/chat.tsx
"use client";

import { useState } from "react";

type Message = { role: "user" | "assistant"; content: string };

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);

  async function send() {
    const question = input.trim();
    if (!question || streaming) return;

    const history: Message[] = [...messages, { role: "user", content: question }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setStreaming(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }),
      });
      if (!response.body) throw new Error("no stream");

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let answer = "";

      // Appended per chunk rather than replaced: the last message is rebuilt
      // each tick, so React re-renders only the tail.
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        answer += value;
        setMessages([...history, { role: "assistant", content: answer }]);
      }
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {messages.map((message, index) => (
        <p key={index} data-role={message.role}>
          {message.content}
        </p>
      ))}
      <input
        value={input}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => event.key === "Enter" && void send()}
        disabled={streaming}
      />
    </div>
  );
}
```

```json path=frontend/package.json
{
  "name": "ai-frontend",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "api:types": "curl -s http://localhost:8000/openapi.json > openapi.json && openapi-typescript openapi.json -o types/api.ts && rm openapi.json"
  },
  "dependencies": {
    "next": "^15.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "openapi-typescript": "^7.0.0",
    "typescript": "^5.0.0"
  }
}
```

## What is missing on purpose

No auth, no persistence, no rate limiting. All three are real requirements and
all three have opinions attached; a starter that picks them for you is a starter
you have to unpick.
