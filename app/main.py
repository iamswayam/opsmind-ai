from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
import io

from app.db import get_connection
from app.chunking import chunk_text
from app.gemini_client import embed_text, generate_answer

app = FastAPI(title="OpsMind AI")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Upload path ----------

def extract_text(file: UploadFile, raw: bytes) -> str:
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # logs / SOPs / api docs uploaded as plain text
    return raw.decode("utf-8", errors="ignore")


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), doc_type: str = Form(...)):
    raw = await file.read()
    text = extract_text(file, raw)
    if not text.strip():
        raise HTTPException(400, "Couldn't extract any text from this file")

    chunks = chunk_text(text)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename, doc_type) VALUES (%s, %s) RETURNING id",
                (file.filename, doc_type),
            )
            document_id = cur.fetchone()["id"]

            for i, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                cur.execute(
                    "INSERT INTO chunks (document_id, content, embedding, chunk_index) "
                    "VALUES (%s, %s, %s, %s)",
                    (document_id, chunk, embedding, i),
                )
        conn.commit()
    finally:
        conn.close()

    return {"document_id": document_id, "filename": file.filename, "chunks_created": len(chunks)}


@app.get("/documents")
def list_documents():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename, doc_type, uploaded_at FROM documents ORDER BY uploaded_at DESC")
            return cur.fetchall()
    finally:
        conn.close()


# ---------- Chat path ----------

class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str


TOP_K = 5


@app.post("/chat")
def chat(req: ChatRequest):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 1. create a conversation if this is a new session
            conversation_id = req.conversation_id
            if conversation_id is None:
                cur.execute(
                    "INSERT INTO conversations (title) VALUES (%s) RETURNING id",
                    (req.question[:60],),
                )
                conversation_id = cur.fetchone()["id"]

            # 2. pull prior turns for this conversation (conversation memory)
            cur.execute(
                "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at",
                (conversation_id,),
            )
            history = cur.fetchall()

            # 3. embed the question and retrieve the most relevant chunks
            question_embedding = embed_text(req.question)
            cur.execute(
                "SELECT c.content, d.filename FROM chunks c "
                "JOIN documents d ON d.id = c.document_id "
                "ORDER BY c.embedding <=> %s LIMIT %s",
                (question_embedding, TOP_K),
            )
            retrieved = cur.fetchall()
            context_chunks = [f"[{r['filename']}] {r['content']}" for r in retrieved]

            # 4. generate the answer
            answer = generate_answer(req.question, context_chunks, history)

            # 5. persist both turns
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
                (conversation_id, req.question),
            )
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'assistant', %s)",
                (conversation_id, answer),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": list({r["filename"] for r in retrieved}),
    }


# TODO (Day 5-6): add a /chat/stream endpoint using StreamingResponse + generate_answer_stream
# TODO (Day 5-6): add an /incidents/summarize endpoint that's more agentic — decides whether
#   to retrieve, ask a clarifying question, or route to a "next troubleshooting step" flow.
#   This is where LangGraph earns its place instead of being a plain RAG call.
