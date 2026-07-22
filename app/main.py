from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
import fitz  # PyMuPDF — used only as a fallback for scanned/image-based PDFs
import io
import os
import json

from app.db import get_connection
from app.chunking import chunk_text
from app.gemini_client import embed_text, generate_answer, generate_answer_stream, ocr_page_image
from google.genai import errors as genai_errors
from psycopg2.extras import Json

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

def extract_pages(file: UploadFile, raw: bytes) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) tuples, 1-indexed.
    Plain text files (logs/SOPs) are treated as a single page (page 1) —
    there's no real pagination concept for those, but keeping the same
    return shape means upload_document doesn't need a separate code path."""
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        if any(text.strip() for _, text in pages):
            return pages

        # Fallback: no text layer found on any page (scanned/image-based PDF).
        # Render each page to a PNG and transcribe it with Gemini vision.
        ocr_pages = []
        pdf_doc = fitz.open(stream=raw, filetype="pdf")
        for i, page in enumerate(pdf_doc):
            pix = page.get_pixmap(dpi=200)  # 200 dpi is enough for clean transcription
            png_bytes = pix.tobytes("png")
            ocr_pages.append((i + 1, ocr_page_image(png_bytes)))
        pdf_doc.close()
        return ocr_pages

    # logs / SOPs / api docs uploaded as plain text — one "page"
    return [(1, raw.decode("utf-8", errors="ignore"))]


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), doc_type: str = Form(...)):
    raw = await file.read()
    pages = extract_pages(file, raw)
    if not any(text.strip() for _, text in pages):
        raise HTTPException(400, "Couldn't extract any text from this file")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename, doc_type) VALUES (%s, %s) RETURNING id",
                (file.filename, doc_type),
            )
            document_id = cur.fetchone()["id"]

            chunk_index = 0
            for page_num, page_text in pages:
                if not page_text.strip():
                    continue
                # chunk each page independently — keeps page metadata accurate per chunk
                # instead of losing page boundaries by chunking the whole document at once
                for chunk in chunk_text(page_text):
                    embedding = embed_text(chunk)
                    cur.execute(
                        "INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (document_id, chunk, embedding, chunk_index, Json({"page": page_num})),
                    )
                    chunk_index += 1
        conn.commit()
    finally:
        conn.close()

    return {"document_id": document_id, "filename": file.filename, "chunks_created": chunk_index}


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


def _fetch_history(cur, conversation_id):
    cur.execute(
        "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at",
        (conversation_id,),
    )
    return cur.fetchall()


def _retrieve_chunks(cur, question_embedding, document_ids):
    query = (
        "SELECT c.content, d.filename, d.id AS document_id, c.metadata, "
        "(c.embedding <=> %s::vector) AS distance "
        "FROM chunks c JOIN documents d ON d.id = c.document_id"
    )
    params = [question_embedding]
    if document_ids:
        query += " WHERE c.document_id = ANY(%s)"
        params.append(document_ids)
    query += " ORDER BY distance LIMIT %s"
    params.append(TOP_K)
    cur.execute(query, params)
    return cur.fetchall()


def _build_sources(retrieved):
    return [
        {
            "filename": r["filename"],
            "snippet": r["content"][:400],
            "page": (r["metadata"] or {}).get("page"),
        }
        for r in retrieved
    ]


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
            history = _fetch_history(cur, conversation_id)

            # 3. embed the question and retrieve the most relevant chunks,
            #    optionally scoped to specific documents
            question_embedding = embed_text(req.question)
            retrieved = _retrieve_chunks(cur, question_embedding, req.document_ids)

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
                    sources = _build_sources(retrieved)
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


# ---------- Streaming chat ----------

def _sse(payload: dict) -> str:
    """Format one Server-Sent Event line. Frontend splits on the blank-line
    separator and parses each 'data: ...' payload as JSON."""
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    conn = get_connection()

    def event_generator():
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

                # 2. history + retrieval — identical to the non-streaming /chat
                history = _fetch_history(cur, conversation_id)
                question_embedding = embed_text(req.question)
                retrieved = _retrieve_chunks(cur, question_embedding, req.document_ids)

                # 3. confidence gate
                if not retrieved or retrieved[0]["distance"] > DISTANCE_THRESHOLD:
                    answer = (
                        "I couldn't find anything in your uploaded documents that addresses this. "
                        "Try rephrasing, or attach a specific document if you know which one it's in."
                    )
                    sources = []
                    yield _sse({"type": "chunk", "text": answer})
                else:
                    context_chunks = [f"[{r['filename']}] {r['content']}" for r in retrieved]
                    sources = _build_sources(retrieved)
                    answer_parts = []
                    try:
                        for piece in generate_answer_stream(req.question, context_chunks, history):
                            answer_parts.append(piece)
                            yield _sse({"type": "chunk", "text": piece})
                    except genai_errors.ClientError as e:
                        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                            print(f"[chat/stream] All models hit quota/limits: {e}")
                            msg = (
                                "This app has hit today's free-tier request limit across every "
                                "available Gemini model. It'll reset within a day — try again later."
                            )
                        else:
                            print(f"[chat/stream] Gemini ClientError: {e}")
                            msg = "Something went wrong talking to Gemini. Check the server logs for details."
                        answer_parts = [msg]
                        sources = []
                        yield _sse({"type": "chunk", "text": msg})
                    except genai_errors.ServerError as e:
                        print(f"[chat/stream] All models returned ServerError: {e}")
                        msg = "Gemini's servers are temporarily overloaded. Please try asking again in a moment."
                        answer_parts = [msg]
                        sources = []
                        yield _sse({"type": "chunk", "text": msg})
                    answer = "".join(answer_parts)

                # 4. persist both turns, same as the non-streaming endpoint
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
                    (conversation_id, req.question),
                )
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'assistant', %s)",
                    (conversation_id, answer),
                )
            conn.commit()

            # 5. final event carries metadata the frontend needs once streaming is done
            yield _sse({"type": "done", "conversation_id": conversation_id, "sources": sources})
        finally:
            conn.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# TODO (Day 3-5): add an /incidents/summarize endpoint that's more agentic — decides whether
#   to retrieve, ask a clarifying question, or route to a "next troubleshooting step" flow.
#   This is where LangGraph earns its place instead of being a plain RAG call.