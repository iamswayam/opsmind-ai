from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
import fitz  # PyMuPDF — used only as a fallback for scanned/image-based PDFs
import io
import os

from app.db import get_connection
from app.chunking import chunk_text
from app.gemini_client import embed_text, generate_answer, ocr_page_image
from google.genai import errors as genai_errors

app = FastAPI(title="OpsMind AI")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Upload path ----------

def extract_text(file: UploadFile, raw: bytes) -> str:
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text

        # Fallback: no text layer found (scanned/image-based PDF).
        # Render each page to a PNG and transcribe it with Gemini vision.
        ocr_pages = []
        pdf_doc = fitz.open(stream=raw, filetype="pdf")
        for page in pdf_doc:
            pix = page.get_pixmap(dpi=200)  # 200 dpi is enough for clean transcription
            png_bytes = pix.tobytes("png")
            ocr_pages.append(ocr_page_image(png_bytes))
        pdf_doc.close()
        return "\n\n".join(ocr_pages)

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
def list_documents(response: Response):
    response.headers["Cache-Control"] = "no-store"
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
    document_ids: list[int] | None = None   # if set, restrict retrieval to these docs only


TOP_K = 5
# Cosine distance from pgvector's <=> operator: 0 = identical, 2 = opposite.
# Above this, the "best" match is too dissimilar to trust — treat as no relevant context.
# Tune this against your own data; it's a heuristic, not a physical constant.
DISTANCE_THRESHOLD = 0.6


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

            # 3. embed the question and retrieve the most relevant chunks,
            #    optionally scoped to specific documents
            question_embedding = embed_text(req.question)

            query = (
                "SELECT c.content, d.filename, d.id AS document_id, "
                "(c.embedding <=> %s::vector) AS distance "
                "FROM chunks c JOIN documents d ON d.id = c.document_id"
            )
            params = [question_embedding]

            if req.document_ids:
                query += " WHERE c.document_id = ANY(%s)"
                params.append(req.document_ids)

            query += " ORDER BY distance LIMIT %s"
            params.append(TOP_K)

            cur.execute(query, params)
            retrieved = cur.fetchall()

            # 4. confidence gate — if even the best match is too dissimilar,
            #    don't let the LLM paper over bad retrieval with a confident-sounding guess
            if not retrieved or retrieved[0]["distance"] > DISTANCE_THRESHOLD:
                answer = (
                    "I couldn't find anything in your uploaded documents that addresses this. "
                    "Try rephrasing, or attach a specific document if you know which one it's in."
                )
                sources = []
            else:
                context_chunks = [f"[{r['filename']}] {r['content']}" for r in retrieved]
                try:
                    answer = generate_answer(req.question, context_chunks, history)
                    sources = [
                        {"filename": r["filename"], "snippet": r["content"][:400]}
                        for r in retrieved
                    ]
                except genai_errors.ClientError as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        print(f"[chat] All models in the fallback chain hit quota/limits: {e}")
                        answer = (
                            "This app has hit today's free-tier request limit across every "
                            "available Gemini model. It'll reset within a day — try again later."
                        )
                    else:
                        print(f"[chat] Gemini ClientError: {e}")
                        answer = "Something went wrong talking to Gemini. Check the server logs for details."
                    sources = []
                except genai_errors.ServerError as e:
                    # every model in the fallback chain was overloaded/unavailable and
                    # retries were already exhausted — fail gracefully instead of a raw 500
                    print(f"[chat] All models in the fallback chain returned ServerError: {e}")
                    answer = (
                        "Gemini's servers are temporarily overloaded. "
                        "Please try asking again in a moment."
                    )
                    sources = []

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
        "sources": sources,
    }


# ---------- Document deletion ----------

@app.delete("/documents/{document_id}")
def delete_document(document_id: int):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # ON DELETE CASCADE on chunks.document_id (see db/init.sql) removes
            # the document's chunks automatically — no separate cleanup query needed.
            cur.execute("DELETE FROM documents WHERE id = %s RETURNING id", (document_id,))
            deleted = cur.fetchone()
            if not deleted:
                raise HTTPException(404, "Document not found")
        conn.commit()
    finally:
        conn.close()
    return {"deleted_id": document_id}


# TODO (Day 5-6): add a /chat/stream endpoint using StreamingResponse + generate_answer_stream
# TODO (Day 5-6): add an /incidents/summarize endpoint that's more agentic — decides whether
#   to retrieve, ask a clarifying question, or route to a "next troubleshooting step" flow.
#   This is where LangGraph earns its place instead of being a plain RAG call.