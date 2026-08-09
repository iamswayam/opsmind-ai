# OpsMind AI Project Journey

This journal documents OpsMind AI stage by stage, using the current project files as the source of truth. It is meant for someone catching up on the whole build, not just scanning a changelog.

## Current Project Map

Root files currently present:

- `.env` - local environment file; should stay untracked and never be printed or committed.
- `.env.example` - template for required environment variables.
- `.gitignore` - ignores secrets, Python caches, virtualenvs, Docker logs, and editor/OS files.
- `Dockerfile` - builds the FastAPI container; copies both `app/` and `tests/`.
- `docker-compose.yml` - runs the API and Postgres/pgvector as separate services; mounts `app/` and `tests/` for live-reload.
- `README.md` - current architecture, setup, endpoint, and lessons-learned guide.
- `requirements.txt` - pinned Python dependencies.

Files under `app/`:

- `app/__init__.py`
- `app/main.py` - HTTP layer: normal chat, `@doc`, `@art`, the agentic endpoint, upload/extraction, document management.
- `app/agent.py` - the LangGraph agentic decision layer: state schema, all nodes, conditional routing, compiled graph.
- `app/retrieval.py` - shared retrieval helpers used by both `main.py` and `agent.py`.
- `app/gemini_client.py` - all Gemini API calls: embeddings (text and image, shared vector space), generation, streaming, structured/JSON output, diagram generation, OCR, and the model fallback chain.
- `app/chunking.py` - paragraph-aware chunking.
- `app/db.py` - connection handling, pgvector type registration.
- `app/static/index.html` - single-file frontend (no framework, no build step).

Files under `tests/`:

- `tests/test_agent.py` - offline test proving the investigate loop's step cap is enforced in code, not just prompted for.

Files under `db/`:

- `db/init.sql` - schema for fresh installs.
- `db/migrations/` - migrations for databases created before a given feature (e.g. multimodal columns) existed.

## Stage 1 - Project Scaffold

### What Was Built

OpsMind AI started as a FastAPI service backed by PostgreSQL with the `pgvector` extension. Docker Compose was added so the API and database run as separate containers on the same Compose network.

The database schema was created with four core tables:

- `documents`: one row per uploaded source file, with `filename`, `doc_type`, and `uploaded_at`.
- `chunks`: extracted text chunks linked to documents, with `content`, `chunk_index`, optional `metadata`, and a `VECTOR(768)` embedding column.
- `conversations`: one row per chat session.
- `messages`: persisted user/assistant turns linked to a conversation.

The `chunks.document_id` foreign key uses `ON DELETE CASCADE`, so deleting a document automatically deletes its embedded chunks.

### Problem It Solved

This gave the project a real RAG foundation: upload documents, split them into chunks, embed those chunks, store them in pgvector, and later retrieve them by similarity during chat.

### Where It Lives

- `Dockerfile`, `docker-compose.yml`, `db/init.sql`, `requirements.txt`, `app/main.py`, `app/db.py`, `app/chunking.py`

## Stage 2 - Gemini API Integration: Embeddings and Chat

### What Was Built

Gemini integration was centralized in `app/gemini_client.py`, using `client.models.embed_content(...)` for embeddings and `client.models.generate_content(...)` (later, a shared fallback helper) for chat and OCR generation. The embedding function requested `output_dimensionality=768` to match the pgvector column.

### Problem It Solved

Gemini's embedding models return more than 768 dimensions by default. Requesting 768 explicitly kept vectors aligned with `VECTOR(768)` without widening the schema — pgvector's IVFFlat index doesn't support vectors over 2000 dimensions, and 768 is sufficient for this use case.

### Current Note

The embedding model used here was originally `gemini-embedding-001` (text-only). It was later replaced with `gemini-embedding-2` in Stage 16 to support multimodal (text + image) embeddings in one shared vector space — see that stage for why.

### Where It Lives

- `app/gemini_client.py`, `db/init.sql`, `app/main.py`

## Stage 3 - pgvector Similarity Search Bug

### What Was Built

The chat endpoint embeds the user's question, then runs cosine similarity against stored chunks using `(c.embedding <=> %s::vector) AS distance` — with an explicit `::vector` cast.

### Problem It Solved

Document upload worked because inserts into a typed `VECTOR(768)` column can use assignment casting. Similarity search failed with an operator error (`operator does not exist: vector <=> numeric[]`) because the `<=>` operator needs operands it can resolve as vectors in the query expression itself — passing a Python list through psycopg2 wasn't enough there. Adding `%s::vector` made the operator resolve correctly.

### Where It Lives

- `app/retrieval.py`, `app/db.py`

## Stage 4 - Gemini API Surface Version Bug

### What Was Built

The Gemini client is created without pinning the API version (`client = genai.Client(api_key=api_key)`), with an explicit code comment warning not to set `http_options` to `api_version: "v1"`.

### Problem It Solved

The older Gemini `v1` REST surface rejected fields such as `systemInstruction`, producing `Unknown name systemInstruction: Cannot find field`. The default `v1beta` surface supports the fields this app uses. This bug regressed more than once during development after being "fixed" in conversation but not actually saved to the file — it's now a standing habit to grep for `api_version` after any Gemini-related edit.

### Where It Lives

- `app/gemini_client.py`

## Stage 5 - Model Deprecation and the Fallback Chain

### What Was Built

Chat, OCR, streaming, and structured-output generation all route through a shared `_generate_with_fallback(...)` helper, which tries an ordered model list (`CHAT_MODEL_FALLBACK_CHAIN`). On a `ServerError`/5xx, it retries the *same* model with backoff first (transient). On a 429 quota error or 404 (unavailable), it moves to the *next* model instead (retrying won't help). Any other client error raises immediately, since fallback would likely hide a real bug.

### Problem It Solved

A single hard-coded chat model became fragile as model availability changed and free-tier quotas turned out to differ significantly by model — as low as 20 requests/day for some models on this project, confirmed directly on the AI Studio quota dashboard rather than trusted from inconsistent blog posts. The fallback chain made model failures survivable without manual intervention.

### Where It Lives

- `app/gemini_client.py`

## Stage 6 - Git Security Incident and Cleanup

### What Was Built

A `.gitignore` excluding `.env`, Python caches/virtualenvs, Docker logs, and editor/OS files.

### Problem It Solved

The real Gemini API key was accidentally committed and pushed, triggering GitHub push protection. Remediation: rotated the key, removed `.env` from Git history with `git-filter-repo`, verified the path no longer appeared anywhere in history, then kept `.env` ignored going forward.

### Where It Lives

- `.gitignore`, `.env.example`

## Stage 7 - Frontend UI

### What Was Built

A single-file frontend in `app/static/index.html`, served directly by FastAPI (`GET /`) — same-origin API calls, no separate frontend server, no CORS setup. Dark ops-console styling; sidebar document upload/list; chat pane; composer with document-attach controls; assistant markdown rendering via `marked.js`.

### Problem It Solved

Moved the project from API-only to an end-to-end usable application, and fixed model output like `**bold**` or lists rendering as raw syntax instead of formatted text.

### Where It Lives

- `app/static/index.html`, `app/main.py`

## Stage 8 - OCR Fallback for Scanned PDFs

### What Was Built

PDF extraction has a two-step path: try `pypdf` text extraction first; if a page returns no usable text, render it to a PNG via PyMuPDF and transcribe it with Gemini vision.

### Problem It Solved

Scanned/image-based PDFs have no real text layer, so `pypdf` returns empty text, causing `Couldn't extract any text from this file`. The fallback indexes scanned PDFs without requiring external OCR binaries.

### Where It Lives

- `app/main.py`, `app/gemini_client.py`, `requirements.txt`

## Stage 9 - Document-Scoped Chat, Deletion, Confidence Gate, and Source Transparency

### What Was Built

`/chat` accepts optional `document_ids` to scope retrieval (`WHERE c.document_id = ANY(%s)`), applies a confidence gate (`DISTANCE_THRESHOLD`) that refuses to generate on weak retrieval, and returns grouped, clickable sources. `DELETE /documents/{id}` was added, cascading to chunks automatically. The frontend gained an attach dropup, removable chips, a delete confirmation modal, and auto-detach on delete.

### Problem It Solved

Global retrieval across multiple documents caused cross-document confusion — the top-K chunks could come from the wrong file. Scoping fixed that. The confidence gate prevents a confident-sounding answer from weak/irrelevant retrieval.

### Where It Lives

- `app/main.py`, `app/static/index.html`, `db/init.sql`

## Stage 10 - Document List Caching Bug Fix

### What Was Built

`Cache-Control: no-store` on the `/documents` response, and `cache: 'no-store'` on the frontend's fetch call.

### Problem It Solved

Newly uploaded documents didn't always appear in the sidebar immediately — the browser was silently caching the `GET /documents` request despite no explicit cache headers.

### Where It Lives

- `app/main.py`, `app/static/index.html`

## Stage 11 - README Overhaul (First Pass)

### What Was Built

`README.md` rewritten to reflect the working application rather than the early scaffold — architecture, setup, UI features, endpoints, schema, deferred work, lessons learned.

### Where It Lives

- `README.md`

## Stage 12 - Streaming Responses

### What Was Built

`generate_answer_stream(...)` was implemented for real (previously a stub), sharing a `_build_prompt(...)` helper with `generate_answer(...)` and trying the same `CHAT_MODEL_FALLBACK_CHAIN`. A new `POST /chat/stream` endpoint returns Server-Sent Events — `{"type": "chunk", "text": ...}` as text arrives, then one `{"type": "done", ...}` event. The frontend reads this via `fetch(...).body.getReader()` and renders the growing markdown answer incrementally.

### Problem It Solved

Previously the UI showed nothing until Gemini finished the entire answer. Streaming makes the app feel responsive instead of making the user wait on a silent pause.

### Where It Lives

- `app/gemini_client.py`, `app/main.py`, `app/static/index.html`

## Stage 13 - Paragraph-Aware Chunking and Page-Level Citations

### What Was Built

`chunk_text(...)` groups whole paragraphs up to a size limit instead of slicing at a fixed character count, falling back to fixed-size splitting only for a single paragraph too large to fit on its own. PDF extraction changed to per-page extraction (`extract_pages`), and every chunk carries `{"page": N}` in its `metadata` column. Sources in the UI show "Page N" when available.

### Problem It Solved

Naive fixed-size chunking could cut a chunk off mid-sentence, hurting retrieval quality. Page metadata gives citations precision beyond just "which file."

### Current Note

Page metadata only applies to documents uploaded after this change (and doesn't apply to `.docx` at all — see Stage 17).

### Where It Lives

- `app/chunking.py`, `app/main.py`, `app/static/index.html`

## Stage 14 - The Agentic Decision Layer (LangGraph)

### What Was Built

The single largest addition to the project. `app/agent.py` defines an `AgentState` (`TypedDict`) and a compiled `StateGraph`:

- **Triage** (`triage_node`) - one structured-output call classifying intent as `direct`, `clarify`, or `investigate`. If `clarify`, the clarifying question is produced in the *same* call, so that path costs exactly one LLM call total.
- **Direct** - retrieves and answers with the same confidence gate as `/chat`.
- **Clarify** - spends zero retrieval calls.
- **Investigate** - a bounded loop (`MAX_INVESTIGATION_STEPS = 3`). Each iteration is one combined structured-output call that both picks the next relevant diagnostic check (from whatever was actually retrieved, not a hardcoded list) and evaluates it, appending to `investigation_steps` so later iterations have memory. The cap is enforced in code, not left to the model to self-limit.
- `evidence_status` is a categorical `Literal["sufficient", "insufficient", "conflicting"]`, deliberately not a numeric confidence float, since there was no evaluation data to calibrate a specific number against.

`app/retrieval.py` was created specifically so `agent.py` could use shared retrieval logic without importing from `main.py` (which would create a circular import once `main.py` imports the agent back). A new `generate_structured(...)` function in `gemini_client.py` requests strict JSON output, routed through the same fallback chain as everything else. A new `POST /chat/agent` endpoint runs the graph - additive alongside `/chat` and `/chat/stream`, neither of which was modified.

### Problem It Solved

Every endpoint before this stage did the same thing regardless of what was actually asked: retrieve, then generate. This adds real branching and state-dependent decisions - the actual distinction between "RAG" and "agentic."

### Bug Hit During Implementation

`ValueError: 'answer' is already being used as a state key` - LangGraph doesn't allow a node's registered name to collide with a state field name. The node named `"answer"` collided with `AgentState.answer`; fixed by renaming the node to `"direct_answer"`.

### Where It Lives

- `app/agent.py`, `app/retrieval.py`, `app/gemini_client.py`, `app/main.py`

## Stage 15 - Proper Test Structure

### What Was Built

A standalone script with manual asserts was replaced with a real pytest test at `tests/test_agent.py` (a proper top-level directory, sibling to `app/`, not nested inside it). The test mocks `generate_structured` so the model always requests another investigation iteration, then asserts the loop still stops at exactly `MAX_INVESTIGATION_STEPS` and is forced to `"escalate"` - zero real API calls, proving the cap is enforced deterministically rather than relying on having observed it work in manual testing.

### Problem It Solved

The investigate loop's step cap had only been exercised in live manual testing, where every test question happened to resolve or request evidence on the first iteration - the cap itself had never actually fired and was unverified in practice.

### Where It Lives

- `tests/test_agent.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`

## Stage 16 - Multimodal RAG

### What Was Built

The text embedding model was switched from `gemini-embedding-001` (text-only) to `gemini-embedding-2`, Google's multimodal embedding model - text and images now share one 768-dim vector space, which is what makes cross-modal retrieval meaningful. New functions: `embed_image(...)` and `caption_image(...)` in `gemini_client.py`. PDF upload now also extracts embedded images (`extract_images` in `main.py`, via PyMuPDF), captions and embeds each one, and stores them as `modality = 'image'` chunks with the raw bytes in a new `image_data` column. Retrieval and generation were updated so retrieved image chunks are passed as actual image content into the Gemini generation call (`build_context_images`), not just their captions.

Safety limits were added deliberately, given this project's history with tight free-tier quotas: `MAX_IMAGES_PER_DOCUMENT = 10` and `MIN_IMAGE_DIMENSION = 100` (skipping tiny icons/logos) - each processed image costs two Gemini calls (caption + embed).

### Problem It Solved

The OCR fallback (Stage 8) only ever converts an image to transcribed text - it's multimodal *input* used to produce a text-only RAG system, not multimodal RAG. This stage adds real cross-modal retrieval.

### Where It Lives

- `app/gemini_client.py`, `app/main.py`, `app/retrieval.py`, `app/static/index.html`, `db/init.sql`, `db/migrations/`

## Stage 17 - DOCX Support

### What Was Built

`extract_docx_text(...)` in `main.py`, using `python-docx`, walks paragraphs *and* table cells. A custom `_paragraph_text_with_breaks(...)` helper walks each paragraph's XML directly to preserve manual line breaks (`<w:br/>`/`<w:cr/>`) - `python-docx`'s default `paragraph.text` silently collapses these, which would destroy an ASCII tree/diagram typed as multiple lines inside one Word paragraph. Legacy `.doc` (pre-2007 binary format) is explicitly rejected with a clear error rather than silently falling through to the old plain-text decode path.

### Problem It Solved

`.docx` is a ZIP archive of XML, not plain text - decoding its raw bytes as UTF-8 (the previous fallback for "anything not a PDF") produced binary/XML garbage that the app would then confidently generate answers about.

### Current Note

`.docx` has no real page boundaries stored in the file - the whole document is treated as one logical page.

### Where It Lives

- `app/main.py`, `requirements.txt`

## Stage 18 - @doc Exact-Answer Mode

### What Was Built

A new command, `@doc <question>`, parsed via `parse_command(...)` in `main.py`. `resolve_doc_mode(...)` retrieves a wide candidate pool (`DOC_MODE_CANDIDATE_K = 10`, not just the top match), then makes one structured-output call whose *only* job is to select which candidate genuinely answers the question - it never generates or rewrites the answer text. The selected chunk's section is reconstructed from a wide window of real stored chunks (`DOC_MODE_MAX_WINDOW = 4`) and trimmed to its actual boundaries using heading detection (`_extract_section_text`), starting *after* the heading's own line so the returned answer never repeats the section title. `@doc` responses return empty `sources` - the answer text already *is* the full source content verbatim, so an expandable "sourced:" chip revealing the same content again would be redundant.

### Problem It Solved

The original design took the single closest chunk by raw cosine distance, gated at a strict threshold. That fails loosely-phrased, holistic questions ("brief me about the whole project"), since such questions often aren't closest-by-embedding to the one section that actually answers them best. A fixed-size reconstruction window also either truncated long answers or bled into neighboring sections. Both were fixed by retrieving widely, letting an LLM judge relevance, and trimming at real section boundaries instead of guessed window sizes.

### Why Not Just Ask the LLM to "Preserve Wording"?

Considered and rejected. An LLM instructed not to paraphrase still drifts over longer content. Returning the actual stored chunk text is the only way to guarantee verbatim preservation - and it costs zero extra Gemini calls for the answer itself.

### Where It Lives

- `app/main.py`, `app/retrieval.py`

## Stage 19 - @art Diagram Mode

### What Was Built

A second command, `@art <topic>`, generates a Mermaid diagram grounded in retrieved document content (`generate_diagram(...)` in `gemini_client.py`), delivered as one complete response - never token-streamed, since a diagram is only valid once its syntax is complete. The frontend detects fenced mermaid code blocks and renders them as real SVG via `mermaid.js`, with a toolbar for "Open full size" and "Download PNG" (canvas-based export, since right-clicking an inline SVG never offers "save as image" in any browser).

### Problem It Solved (First Pass)

Diagrams needed to exist at all, and needed to be genuinely readable and exportable, not just described in text.

### Problem It Solved (Second Pass - Layout)

An early version rendered diagrams that sprawled excessively wide. Increasing render scale only magnified the bad layout. Root cause: the generation prompt gave the model no layout constraints, so given rich content it naturally produced many sibling nodes side-by-side at each rank. Fixed at the generation level (max 3 siblings per rank before requiring a subgraph, cluster related components, cap node count, keep labels short), combined with adaptive client-side sizing that scales small diagrams up but caps large ones near their natural size instead of blowing them up further.

### Where It Lives

- `app/gemini_client.py`, `app/main.py`, `app/static/index.html`

## Stage 20 - @ Command Autocomplete and Styling

### What Was Built

Typing `@` in the composer shows a suggestion dropup (`AVAILABLE_COMMANDS`, an extensible array - adding a future command means adding one entry) with keyboard navigation and click-to-select. The `@doc`/`@art` command token renders in a distinct blue, live, as the user types.

### Implementation Note

A native `<textarea>` cannot style part of its own text. This was implemented with a "highlight overlay": a styled backdrop `<div>` positioned exactly behind the textarea, mirroring its content with the command portion wrapped in a colored `<span>`, while the textarea itself has transparent text and a visible caret. Both layers require identical font, size, line-height, and padding, and their scroll position has to be kept in sync manually.

### Where It Lives

- `app/static/index.html`

## Current Endpoint Reference

| Method | Path | Signature / Request Shape | Response Purpose |
|---|---|---|---|
| `GET` | `/` | No request body | Serves `app/static/index.html`. |
| `GET` | `/health` | No request body | Returns `{"status": "ok"}`. |
| `POST` | `/documents/upload` | `multipart/form-data` with `file: UploadFile` and `doc_type: str` | Extracts text (PDF/DOCX/OCR) and embedded images, chunks per page/section, embeds both modalities, stores the document. |
| `GET` | `/documents` | No request body | Returns all documents, newest first; `Cache-Control: no-store`. |
| `POST` | `/chat` | JSON: `conversation_id?: int`, `question: str`, `document_ids?: list[int]` | Normal chat, `@doc`, and `@art` modes all route through here based on a parsed command prefix. |
| `POST` | `/chat/stream` | Same shape as `/chat` | Same logic, streamed via Server-Sent Events. |
| `POST` | `/chat/agent` | Same shape as `/chat` | Runs the LangGraph agent; response also includes `intent` and `investigation_steps`. |
| `DELETE` | `/documents/{document_id}` | Path parameter `document_id: int` | Deletes the document (cascades to chunks); 404 if missing. |

## Current Database Reference

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id           SERIAL PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,
    embedding    VECTOR(768) NOT NULL,
    chunk_index  INTEGER NOT NULL,
    metadata     JSONB DEFAULT '{}',
    modality     TEXT NOT NULL DEFAULT 'text',
    image_data   BYTEA
);

CREATE INDEX chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE conversations (
    id         SERIAL PRIMARY KEY,
    title      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`modality` and `image_data` were added after `chunks` already existed in some deployments - see `db/migrations/` for the corresponding `ALTER TABLE` statements, since `db/init.sql` only runs on a database's first boot.

## Current Implementation Notes

- `app/db.py` opens a fresh psycopg2 connection per call, using `RealDictCursor`, with pgvector's Python type registered.
- `app/chunking.py` groups whole paragraphs up to `chunk_size=800`, falling back to fixed-size splitting only when a single paragraph exceeds that on its own.
- `app/gemini_client.py` requires `GEMINI_API_KEY` at import time and centralizes every Gemini call through `_generate_with_fallback`.
- `app/retrieval.py` is imported by both `app/main.py` and `app/agent.py`; nothing in either duplicates retrieval SQL.
- `app/main.py`'s `/chat` and `/chat/stream` share history/retrieval logic and branch on a parsed command (`normal` / `doc` / `art`) before generation.
- `app/agent.py`'s `/chat/agent` is fully separate from `/chat` - additive, not a replacement - and not yet wired into the frontend.
- The frontend is framework-free, using CDN-loaded `marked.js` and `mermaid.js`, with a custom highlight-overlay input for command styling.
- `tests/test_agent.py` is the project's only automated test so far.

## Deferred Work

- Wire `/chat/agent` into the frontend UI (currently Swagger-only).
- Hybrid search (vector + full-text via `tsvector`) for exact identifiers pure embeddings handle poorly.
- A RAG evaluation dashboard - golden Q&A set, tracked retrieval/answer-quality metrics.
- Auth, structured logging.
