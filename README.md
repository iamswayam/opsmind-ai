# OpsMind AI

Internal copilot for support/ops teams: upload SOPs, PDFs, logs, and incident
reports, then ask questions grounded in that content via RAG — "why did this
fail," "show the relevant SOP," "summarize this incident" — with the answer
backed by the actual uploaded documents, not the model's general knowledge.

**Status:** upload → chunk → embed → store and chat → retrieve → generate →
store both work end to end. Streaming and the agentic decision layer (see
below) are the next build phase.

## Stack

FastAPI · PostgreSQL + pgvector · Gemini API (embeddings + generation) · Docker

## Run it

1. Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) — no card required.
2. `cp .env.example .env` and paste your key into `GEMINI_API_KEY`.
3. `docker compose up --build`
4. API is live at **http://localhost:8000/docs** — FastAPI's interactive Swagger
   UI, test uploads and chat directly from the browser, no frontend needed yet.

## Architecture

**Upload path:** file → extract text (PDF via `pypdf`, plain text for logs/SOPs)
→ chunk (fixed-size, overlapping) → embed each chunk with Gemini → store in
`chunks` (pgvector column) alongside the source document reference.

**Chat path:** question → embed the question → cosine-similarity search against
`chunks` (`ORDER BY embedding <=> question_embedding::vector`) → top 5 matches +
prior conversation history sent to Gemini as context → answer generated,
grounded in retrieved content → both turns persisted to `messages`.

## Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/documents/upload` | POST | Upload a file (`multipart/form-data`, fields: `file`, `doc_type`) — extracts, chunks, embeds, stores |
| `/documents` | GET | List all uploaded documents |
| `/chat` | POST | `{"question": "...", "conversation_id": optional}` — retrieves relevant context and returns a grounded answer |
| `/health` | GET | Basic liveness check |

## Database schema

- `documents` — one row per uploaded file (filename, doc_type, uploaded_at)
- `chunks` — chunked content + `vector(768)` embedding + reference to source document
- `conversations` — one row per chat session
- `messages` — full turn-by-turn history per conversation, so follow-up questions have context

Full schema lives in `db/init.sql` and runs automatically on first container boot.

## What's deliberately left for you to build (see TODOs in the code)

- **Streaming** (`generate_answer_stream` in `gemini_client.py`) — get the
  non-streaming version working and tested first; streaming is the same prompt
  wrapped differently, not a separate concept.
- **Better chunking** (`chunking.py`) — naive fixed-size chunking works, but
  compare it against paragraph/section-aware splitting once the pipeline runs.
  This comparison is a genuinely good interview story.
- **The agentic layer** — right now `/chat` always retrieves. A LangGraph agent
  that *decides* whether to retrieve, ask a clarifying question, or walk
  through a troubleshooting flow step-by-step is the difference between "a RAG
  demo" and "OpsMind." That's the next phase of this build.
- **Auth, rate limiting, structured logging/evals** — add once the agentic layer works.

## Why this structure

- `app/db.py` — one place to open connections, pgvector's Python type registered
  so vectors pass through as plain Python lists
- `app/gemini_client.py` — all LLM calls isolated here, so swapping models or
  adding retries/evals later doesn't touch endpoint code
- `app/chunking.py` — isolated so you can swap strategies without touching ingestion logic
- `app/main.py` — thin HTTP layer, the actual RAG logic is readable top to bottom in `/chat`

## Lessons learned / troubleshooting notes

A few real issues hit during this build, kept here since they're useful context
for anyone (including future me) extending this:

- **pgvector `<=>` operator needs an explicit cast.** `psycopg2` + `register_vector`
  handles the *insert* path fine via an assignment cast, but the `<=>` similarity
  operator only resolves implicit casts — so the query needs `%s::vector`
  explicitly, or it fails with `operator does not exist: vector <=> numeric[]`.
- **`gemini-embedding-001` returns 3072-dim vectors by default**, truncated to
  768 via `output_dimensionality` in `EmbedContentConfig` to match the pgvector
  column (kept at 768 since pgvector's IVFFlat index doesn't support >2000 dims,
  and 768 is sufficient for this use case).
- **Don't pin the Gemini client to `api_version: "v1"`.** The older `v1` REST
  surface doesn't recognize `systemInstruction` (or `responseMimeType`/
  `responseSchema`) — those require the default `v1beta` surface.
- **Pin chat models to an alias, not a dated string.** `gemini-2.5-flash` was
  pulled from new-user access ahead of its published deprecation date. Using
  `gemini-flash-latest` avoids hard-coding a model string that can disappear
  without much warning.