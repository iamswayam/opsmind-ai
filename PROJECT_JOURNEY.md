# OpsMind AI Project Journey

This journal documents OpsMind AI stage by stage, using the current project files as the source of truth. It is meant for someone catching up on the whole build, not just scanning a changelog.

## Current Project Map

Root files currently present:

- `.env` - local environment file; should stay untracked and never be printed or committed.
- `.env.example` - template for required environment variables.
- `.gitignore` - ignores secrets, Python caches, virtualenvs, Docker logs, and editor/OS files.
- `Dockerfile` - builds the FastAPI container.
- `docker-compose.yml` - runs the API and Postgres/pgvector as separate services.
- `README.md` - current architecture, setup, endpoint, and lessons-learned guide.
- `requirements.txt` - pinned Python dependencies.

Files under `app/`:

- `app/__init__.py`
- `app/chunking.py`
- `app/db.py`
- `app/gemini_client.py`
- `app/main.py`
- `app/static/index.html`

Files under `db/`:

- `db/init.sql`

## Stage 1 - Project Scaffold

### What Was Built

OpsMind AI started as a FastAPI service backed by PostgreSQL with the `pgvector` extension. Docker Compose was added so the API and database run as separate containers on the same Compose network.

The database schema was created with four core tables:

- `documents`: one row per uploaded source file, with `filename`, `doc_type`, and `uploaded_at`.
- `chunks`: extracted text chunks linked to documents, with `content`, `chunk_index`, optional `metadata`, and a `VECTOR(768)` embedding column.
- `conversations`: one row per chat session.
- `messages`: persisted user/assistant turns linked to a conversation.

The `chunks.document_id` foreign key uses `ON DELETE CASCADE`, so deleting a document automatically deletes its embedded chunks. The schema also defines an IVFFlat cosine index on `chunks.embedding`.

### Problem It Solved

This gave the project a real RAG foundation: upload documents, split them into chunks, embed those chunks, store them in pgvector, and later retrieve them by similarity during chat.

### Where It Lives

- `Dockerfile`
- `docker-compose.yml`
- `db/init.sql`
- `requirements.txt`
- `app/main.py`
- `app/db.py`
- `app/chunking.py`

## Stage 2 - Gemini API Integration: Embeddings and Chat

### What Was Built

Gemini integration was centralized in `app/gemini_client.py`. The current code uses:

- `EMBED_MODEL = "gemini-embedding-001"`
- `client.models.embed_content(...)` for embeddings.
- `client.models.generate_content(...)` through a shared fallback helper for chat and OCR generation.

The embedding function currently requests:

```python
types.EmbedContentConfig(output_dimensionality=768)
```

That keeps returned vectors aligned with the database column:

```sql
embedding VECTOR(768) NOT NULL
```

The chat path uses `generate_answer(question, context_chunks, history)`, with a system instruction that requires answers to use only retrieved context and avoid inline filename prefaces because the UI renders sources separately.

### Problem It Solved

Gemini's `gemini-embedding-001` returns 3072-dimensional vectors by default, while this project intentionally stores 768-dimensional vectors. Keeping `VECTOR(768)` avoids widening the column beyond what is useful here and stays compatible with pgvector's IVFFlat indexing limits. The fix was to request 768 dimensions from Gemini instead of changing the schema.

### Current Note

`db/init.sql` still has an older comment saying Gemini's `text-embedding-004` outputs 768-dimensional vectors. The live code now uses `gemini-embedding-001` with explicit `output_dimensionality=768`; the schema itself is still correct.

### Where It Lives

- `app/gemini_client.py`
- `db/init.sql`
- `app/main.py`

## Stage 3 - pgvector Similarity Search Bug

### What Was Built

The `/chat` endpoint embeds the user's question, then runs cosine similarity against stored chunks:

```sql
(c.embedding <=> %s::vector) AS distance
```

The explicit `::vector` cast is present in the current query.

### Problem It Solved

Document upload worked because inserts into a typed `VECTOR(768)` column can use assignment casting. Similarity search failed with an operator error because the `<=>` operator needs operands it can resolve as vectors in the query expression itself. Passing a Python list through psycopg2 was not enough there.

Adding `%s::vector` made the operator resolve correctly.

### Where It Lives

- `app/main.py`
- `app/db.py`

## Stage 4 - Gemini API Surface Version Bug

### What Was Built

The Gemini client is now created without pinning the API version:

```python
client = genai.Client(api_key=api_key)
```

There is also an explicit warning comment in the file not to set `http_options` to `api_version: "v1"`.

### Problem It Solved

The older Gemini `v1` REST surface rejected fields such as `systemInstruction`, producing errors like:

```text
Unknown name systemInstruction: Cannot find field
```

The default `v1beta` surface supports the fields this app uses, including `system_instruction` through `types.GenerateContentConfig`.

### Where It Lives

- `app/gemini_client.py`

## Stage 5 - Model Deprecation and the Fallback Chain

### What Was Built

Chat and OCR generation now route through `_generate_with_fallback(...)`, which tries an ordered model list:

```python
CHAT_MODEL_FALLBACK_CHAIN = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
]
```

The helper behavior is:

- On `ServerError` / 5xx: retry the same model up to three attempts with backoff.
- On 429 quota exhaustion: move to the next model.
- On 404 or `NOT_FOUND`: move to the next model.
- On other client errors: raise immediately, because fallback would likely hide a real request/configuration bug.
- If every model fails: raise the last error.

The helper is used by both:

- `generate_answer(...)`
- `ocr_page_image(...)`

### Problem It Solved

A single hard-coded chat model became fragile when model availability changed and when free-tier quotas differed by model. The fallback chain made model failures survivable: quota and unavailability errors shift to the next model, while transient server errors get retried on the current model first.

### Where It Lives

- `app/gemini_client.py`
- `app/main.py`

## Stage 6 - Git Security Incident and Cleanup

### What Was Built

The project now includes a `.gitignore` that excludes:

- `.env`
- Python cache files and virtualenvs.
- Docker logs.
- OS/editor files such as `.DS_Store`, `.vscode/`, and `.idea/`.

### Problem It Solved

An actual `.env` file exists locally and contains runtime secrets. The project history included a security incident where the real Gemini API key was committed and pushed, triggering GitHub push protection. The remediation described in the project history was to rotate the key, remove `.env` from Git history with `git-filter-repo`, verify the path no longer appeared in history, and then keep `.env` ignored going forward.

The current working tree is clean, and `.gitignore` now protects the local `.env` file from normal future commits.

### Where It Lives

- `.gitignore`
- `.env.example`
- `.env` locally, but intentionally not for commit.

## Stage 7 - Frontend UI

### What Was Built

A single-file frontend was built in `app/static/index.html` and served directly by FastAPI:

- `GET /` returns `app/static/index.html`.
- Static assets are mounted at `/assets` from `app/static`.
- The frontend uses same-origin API calls, so there is no separate frontend server and no CORS setup.

The UI currently includes:

- A dark ops-console style.
- Sidebar document upload and document list.
- A document type selector.
- Chat pane with a conversation title and connection status.
- Composer with document attach controls.
- Assistant markdown rendering through `marked.js`.
- Toasts and upload/delete feedback.

### Problem It Solved

The project moved from API-only behavior to an end-to-end usable application. Users can upload documents, ask questions, attach specific documents, inspect sources, and delete documents from one browser page.

Markdown rendering also fixed the issue where model output like `**bold**` or lists appeared as raw syntax instead of formatted chat text.

### Where It Lives

- `app/static/index.html`
- `app/main.py`

## Stage 8 - OCR Fallback for Scanned PDFs

### What Was Built

PDF extraction now has a two-step path in `extract_text(...)`:

1. Try `pypdf.PdfReader(...).pages[].extract_text()` for PDFs with a text layer.
2. If that returns no usable text, open the PDF with PyMuPDF, render each page to a PNG at 200 DPI, and send each image to Gemini vision through `ocr_page_image(...)`.

The OCR prompt asks Gemini to transcribe readable text in reading order and return only the transcription.

### Problem It Solved

Scanned or image-based PDFs do not have a normal text layer, so `pypdf` returns empty text. Before this stage, those uploads failed with:

```text
Couldn't extract any text from this file
```

The fallback allows scanned PDFs to be indexed without requiring external OCR binaries.

### Where It Lives

- `app/main.py`
- `app/gemini_client.py`
- `requirements.txt`

## Stage 9 - Document-Scoped Chat, Deletion, Confidence Gate, and Source Transparency

### What Was Built

The chat request model now accepts:

```python
class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str
    document_ids: list[int] | None = None
```

The `/chat` endpoint now:

- Creates a conversation if `conversation_id` is absent.
- Loads prior turns from `messages` for the conversation.
- Embeds the question.
- Retrieves top matches from `chunks`.
- Optionally filters retrieval with `WHERE c.document_id = ANY(%s)` when `document_ids` is supplied.
- Orders by cosine distance and limits to `TOP_K = 5`.
- Applies `DISTANCE_THRESHOLD = 0.6`.
- Persists both the user question and assistant answer.
- Returns `conversation_id`, `answer`, and `sources`.

Document deletion was added:

```http
DELETE /documents/{document_id}
```

It deletes the row from `documents`; related chunks are removed by the database cascade.

The frontend now includes:

- A `+` attach dropup above the composer.
- Removable attached-document chips.
- A delete confirmation modal.
- Auto-detach if a deleted document was attached.
- Grouped clickable source tags, where multiple retrieved snippets from the same filename are shown under one expandable tag.

### Problem It Solved

Global retrieval became confusing once multiple documents were uploaded: top-K chunks could come from the wrong file. Document scoping lets the user constrain retrieval to the documents they mean.

The confidence gate prevents weak retrieval from becoming a confident-sounding answer. If no chunk is close enough, the API returns an honest "I couldn't find anything" response instead of sending irrelevant context to Gemini.

Deletion completed basic document lifecycle management, and grouped source tags made answers easier to verify without repeating the same filename several times.

### Where It Lives

- `app/main.py`
- `app/static/index.html`
- `db/init.sql`

## Stage 10 - Document List Caching Bug Fix

### What Was Built

The document list endpoint now sets:

```python
response.headers["Cache-Control"] = "no-store"
```

The frontend also fetches the document list with:

```javascript
fetch(`${API}/documents`, { cache: 'no-store' })
```

### Problem It Solved

After uploads, newly indexed documents did not always appear in the sidebar immediately. The cause was browser caching on `GET /documents`. Adding no-store behavior on both the server response and frontend fetch made the sidebar refresh reliably after upload and deletion.

### Where It Lives

- `app/main.py`
- `app/static/index.html`

## Stage 11 - README Overhaul

### What Was Built

`README.md` was rewritten to reflect the current application rather than just the early scaffold. It now covers:

- What OpsMind AI does.
- How to run it.
- The architecture.
- UI features.
- Endpoint list.
- Database schema.
- Deliberately deferred work.
- Lessons learned and troubleshooting notes.

### Problem It Solved

The README became a practical handoff document. It explains not only what exists, but why some decisions were made: vector dimensionality, pgvector casting, Gemini API versioning, model fallback, document scoping, confidence gating, and frontend caching.

### Where It Lives

- `README.md`

## Current Endpoint Reference

These endpoints are currently defined in `app/main.py`:

| Method | Path | Signature / Request Shape | Response Purpose |
|---|---|---|---|
| `GET` | `/` | No request body | Serves `app/static/index.html`. |
| `GET` | `/health` | No request body | Returns `{"status": "ok"}`. |
| `POST` | `/documents/upload` | `multipart/form-data` with `file: UploadFile` and `doc_type: str` | Extracts text, chunks, embeds, stores a document, and returns `document_id`, `filename`, and `chunks_created`. |
| `GET` | `/documents` | No request body | Returns all documents ordered by newest upload first; sets `Cache-Control: no-store`. |
| `POST` | `/chat` | JSON: `conversation_id?: int`, `question: str`, `document_ids?: list[int]` | Retrieves relevant chunks, optionally scoped to documents, generates or refuses an answer, persists both turns, and returns `conversation_id`, `answer`, and `sources`. |
| `DELETE` | `/documents/{document_id}` | Path parameter `document_id: int` | Deletes the document if found and returns `{"deleted_id": document_id}`; returns 404 if missing. |

## Current Database Reference

The current schema in `db/init.sql` is:

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
    metadata     JSONB DEFAULT '{}'
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

## Current Implementation Notes

- `app/db.py` opens a fresh psycopg2 connection using `DATABASE_URL`, defaults to `postgresql://postgres:devpass@localhost:5432/opsmind`, uses `RealDictCursor`, and registers pgvector on the connection.
- `app/chunking.py` uses naive fixed-size character chunking: `chunk_size=800`, `overlap=100`.
- `app/gemini_client.py` requires `GEMINI_API_KEY` at import time. If the variable is missing, the app raises a `ValueError`.
- `app/main.py` imports `generate_answer_stream` only as a future TODO in comments; the streaming function exists in `app/gemini_client.py` but intentionally raises `NotImplementedError`.
- The current API already stores conversation history in `messages` and passes prior turns into `generate_answer(...)`.
- The current UI is framework-free and depends on CDN-loaded `marked.js` for markdown rendering.

## Deferred Work

The code currently calls out these next stages:

- Add `/chat/stream` with `StreamingResponse` and `generate_answer_stream(...)`.
- Add a more agentic incident/troubleshooting endpoint that can decide whether to retrieve, ask a clarifying question, or route to next steps.
- Improve chunking beyond fixed-size character windows once the end-to-end pipeline has enough real examples to compare.
