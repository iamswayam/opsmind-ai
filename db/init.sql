-- OpsMind AI schema
-- Runs automatically the first time the pgvector container starts.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded file (SOP, PDF, log, API doc, incident report)
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    doc_type    TEXT NOT NULL,   -- 'sop' | 'pdf' | 'log' | 'api_doc' | 'incident'
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunked + embedded content. This is what similarity search runs against.
-- Both text and image chunks live in this table, sharing ONE vector space
-- via gemini-embedding-2 (Google's multimodal embedding model) — that's
-- what makes cross-modal retrieval possible: a text question's embedding
-- can be genuinely close to an image's embedding, not just to other text.
CREATE TABLE chunks (
    id           SERIAL PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,        -- chunk text, OR an image's generated caption
    embedding    VECTOR(768) NOT NULL,
    chunk_index  INTEGER NOT NULL,       -- position within the source doc, useful for context ordering
    metadata     JSONB DEFAULT '{}',    -- e.g. {"page": 3, "section": "Rollback steps"}
    modality     TEXT NOT NULL DEFAULT 'text',  -- 'text' | 'image'
    image_data   BYTEA                  -- raw image bytes, only set when modality = 'image'
);

-- IVFFlat index for fast approximate nearest-neighbor search.
-- lists = rows/1000 is a common starting heuristic; fine to skip this until you have real data volume.
CREATE INDEX chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- One row per chat session
CREATE TABLE conversations (
    id         SERIAL PRIMARY KEY,
    title      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Full message history per conversation, so the agent has memory across turns
CREATE TABLE messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,   -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
