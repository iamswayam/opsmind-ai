# OpsMind AI

Internal copilot for support/ops teams: upload SOPs, PDFs, logs, and incident
reports, then ask questions grounded in that content via RAG — "why did this
fail," "show the relevant SOP," "summarize this incident" — with the answer
backed by the actual uploaded documents, not the model's general knowledge.

**Status:** working end to end, with a real UI. Upload, chunk, embed, store,
retrieve, generate, and display all work — including document-scoped chat,
a confidence gate that refuses to guess on weak retrieval, source
transparency, document deletion, and automatic fallback across Gemini models
when one is rate-limited. Streaming and the agentic decision layer are the
next build phase.

## Stack

FastAPI · PostgreSQL + pgvector · Gemini API (embeddings + generation) · Docker
· vanilla HTML/CSS/JS frontend (no framework — served directly by FastAPI)

## Run it

1. Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) — no card required.
2. `cp .env.example .env` and paste your key into `GEMINI_API_KEY`.
3. `docker compose up --build`
4. Open **http://localhost:8000/** for the chat UI. Swagger docs are still at
   **http://localhost:8000/docs** if you want to hit endpoints directly.

## Architecture

**Upload path:** file → extract text (`pypdf` for text-layer PDFs, `PyMuPDF` +
Gemini vision OCR as a fallback for scanned/image-based PDFs, plain text for
logs/SOPs) → chunk (fixed-size, overlapping) → embed each chunk with Gemini →
store in `chunks` (pgvector column) alongside the source document reference.

**Chat path:** question → optionally scoped to specific attached document(s)
→ embed the question → cosine-similarity search against `chunks`
(`ORDER BY embedding <=> question_embedding::vector`, filtered by
`document_id` if docs are attached) → **confidence gate**: if even the best
match is too dissimilar (cosine distance > `DISTANCE_THRESHOLD`), skip
generation and say so honestly instead of letting the model guess → otherwise,
top 5 matches + prior conversation history sent to Gemini → answer generated
→ both turns persisted to `messages`, with retrieved snippets returned as
clickable, expandable sources in the UI.

**Model resilience:** every Gemini call routes through a fallback chain
(`CHAT_MODEL_FALLBACK_CHAIN` in `gemini_client.py`), ordered by actual daily
quota headroom on the free tier. If one model is rate-limited (429) or
unavailable (404), the next model in the chain is tried automatically instead
of the request failing.

## UI features

- **Document-scoped chat** — click `+` above the composer to attach one or
  more specific documents; chat retrieval is then filtered to only those
  docs' chunks, instead of searching everything (this is what fixes
  cross-document confusion once you have more than one doc uploaded)
- **Delete documents** — trash icon per document in the sidebar, with a
  confirmation modal before deleting; cascades to the document's chunks
  automatically (`ON DELETE CASCADE`) and auto-detaches it from chat if it
  was currently attached
- **Clickable sources** — each `sourced:` tag under an answer expands to show
  the actual retrieved chunk text, so answers are verifiable, not just
  trusted
- **Markdown rendering** — assistant responses render bold/lists/etc.
  properly via `marked.js` instead of showing raw `**`/`###` syntax

## Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Serves the chat UI |
| `/documents/upload` | POST | Upload a file (`multipart/form-data`, fields: `file`, `doc_type`) — extracts (with OCR fallback), chunks, embeds, stores |
| `/documents` | GET | List all uploaded documents |
| `/documents/{id}` | DELETE | Delete a document and its chunks (cascades automatically) |
| `/chat` | POST | `{"question": "...", "conversation_id": optional, "document_ids": optional}` — retrieves relevant context (optionally scoped) and returns a grounded answer with source snippets |
| `/health` | GET | Basic liveness check |

## Database schema

- `documents` — one row per uploaded file (filename, doc_type, uploaded_at)
- `chunks` — chunked content + `vector(768)` embedding + reference to source document (`ON DELETE CASCADE`)
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
- **The agentic layer** — right now `/chat` always retrieves (when a doc is
  relevant enough to pass the confidence gate). A LangGraph agent that
  *decides* whether to retrieve, ask a clarifying question, or walk through a
  troubleshooting flow step-by-step is the difference between "a RAG demo"
  and "OpsMind." That's the next phase of this build.
- **Auth, structured logging/evals dashboard** — add once the agentic layer works.

## Why this structure

- `app/db.py` — one place to open connections, pgvector's Python type registered
  so vectors pass through as plain Python lists
- `app/gemini_client.py` — all LLM calls isolated here, including the model
  fallback chain, so swapping models or adjusting retry behavior never touches
  endpoint code
- `app/chunking.py` — isolated so you can swap strategies without touching ingestion logic
- `app/main.py` — thin HTTP layer; the RAG logic (retrieval, confidence gate,
  generation) is readable top to bottom in `/chat`
- `app/static/index.html` — single-file frontend (HTML/CSS/JS, no build step)
  served directly by FastAPI's `StaticFiles`, so there's no separate frontend
  server or CORS setup to manage

## Lessons learned / troubleshooting notes

Real issues hit during this build, kept here since they're useful context for
anyone (including future me) extending this — several of these turned into
the most interesting parts of the project to talk about:

**Database / retrieval**
- **pgvector `<=>` operator needs an explicit cast.** `psycopg2` + `register_vector`
  handles the *insert* path fine via an assignment cast, but the `<=>` similarity
  operator only resolves implicit casts — so the query needs `%s::vector`
  explicitly, or it fails with `operator does not exist: vector <=> numeric[]`.
- **Global retrieval across all documents causes cross-document confusion**
  once you have more than one doc uploaded — the top-K chunks can come from
  the wrong file entirely. Fixed by adding optional `document_ids` scoping to
  `/chat`, which adds a `WHERE c.document_id = ANY(%s)` filter before the
  similarity ordering.
- **Confident-sounding answers from weak retrieval are worse than no answer.**
  Added a confidence gate: if the top match's cosine distance exceeds a
  threshold, the app says it found nothing relevant instead of letting Gemini
  generate a plausible-sounding answer from irrelevant chunks.

**Gemini API / model config**
- **`gemini-embedding-001` returns 3072-dim vectors by default**, truncated to
  768 via `output_dimensionality` in `EmbedContentConfig` to match the pgvector
  column (kept at 768 since pgvector's IVFFlat index doesn't support >2000 dims,
  and 768 is sufficient for this use case).
- **Don't pin the Gemini client to `api_version: "v1"`.** The older `v1` REST
  surface doesn't recognize `systemInstruction` (or `responseMimeType`/
  `responseSchema`) — those require the default `v1beta` surface. This one
  regressed twice during development after being "fixed" in chat but not
  actually saved to the file — worth double-checking with
  `findstr /n "api_version" app\gemini_client.py` after any Gemini-related edit.
- **Hard-coded dated model strings get cut off without much warning.**
  `gemini-2.5-flash` was pulled from new-user access ahead of its published
  deprecation date. Switching to an alias like `gemini-flash-latest` helps,
  but aliases can silently point to a model with a much stricter free-tier
  quota than expected (see below) — so this alone isn't a complete fix.
- **Free-tier daily quotas are per (project, model), and can be surprisingly
  low** — as low as 20 requests/day per model on this project, verified
  directly on the AI Studio quota dashboard rather than trusted from
  inconsistent third-party blog posts. The real fix was a **model fallback
  chain** (`CHAT_MODEL_FALLBACK_CHAIN` in `gemini_client.py`): every
  `generate_content` call tries models in priority order by actual daily
  headroom, automatically moving to the next model on a 429 (quota) or 404
  (unavailable) instead of failing the request. 5xx server errors get retried
  on the *same* model with backoff first, since those are usually transient;
  quota/availability errors move to the *next* model instead, since retrying
  those won't help.
- **`generate_content` and `embed_content` draw from separate quota pools** —
  embedding calls didn't contribute to the chat quota exhaustion seen during
  heavy upload testing.

**Frontend**
- **`GET` requests can be silently browser-cached** even with no explicit
  cache headers, causing newly uploaded documents to not appear in the
  sidebar immediately. Fixed with `cache: 'no-store'` on the fetch call and a
  `Cache-Control: no-store` response header on `/documents`.
- **Deduping retrieved sources by filename matters.** Returning one source
  entry per retrieved chunk (rather than per unique document) caused the same
  filename to render multiple times in a row under an answer. Sources are now
  grouped by filename, with all matching snippets available under one
  expandable tag.