# OpsMind AI

Internal copilot for support/ops teams: upload SOPs, PDFs, logs, incident
reports, and ask questions grounded in that content via RAG.

## Run it

1. Get a free Gemini API key at aistudio.google.com, no card required.
2. `cp .env.example .env` and paste your key in.
3. `docker compose up --build`
4. API is live at http://localhost:8000/docs (FastAPI's interactive Swagger UI —
   test uploads and chat directly from the browser, no frontend needed yet).

## What's already wired

- Postgres + pgvector running in Docker, schema auto-created on first boot (`db/init.sql`)
- `/documents/upload` — extracts text (PDF or plain text), chunks it, embeds each
  chunk with Gemini, stores it in `chunks`
- `/chat` — embeds the question, does cosine-similarity search against `chunks`
  (`ORDER BY embedding <=> question_embedding`), sends the top 5 matches + prior
  conversation history to Gemini, stores both turns
- `/documents` — list what's been uploaded

## What's deliberately left for you to build (see TODOs in the code)

- **Streaming** (`generate_answer_stream` in `gemini_client.py`) — get the
  non-streaming version working and tested first, streaming is the same prompt
  wrapped differently
- **Better chunking** (`chunking.py`) — naive fixed-size chunking works, but
  compare it against paragraph/section-aware splitting once the pipeline runs;
  this comparison is a genuinely good interview story
- **The agentic layer** — right now `/chat` always retrieves. A LangGraph agent
  that *decides* whether to retrieve, ask a clarifying question, or walk
  through a troubleshooting flow step-by-step is the difference between "a RAG
  demo" and "OpsMind." That's Day 5-6 work.
- **Auth, rate limiting, structured logging/evals** — add once the core loop works.

## Why this structure

- `app/db.py` — one place to open connections, pgvector's Python type registered
  so vectors pass through as plain Python lists
- `app/gemini_client.py` — all LLM calls isolated here, so swapping models or
  adding retries/evals later doesn't touch endpoint code
- `app/chunking.py` — isolated so you can swap strategies without touching ingestion logic
- `app/main.py` — thin HTTP layer, the actual RAG logic is readable top to bottom in `/chat`
