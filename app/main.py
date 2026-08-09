from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
import docx  # python-docx — for .docx extraction
import fitz  # PyMuPDF — used only as a fallback for scanned/image-based PDFs
import io
import os
import json
import re

from app.db import get_connection
from app.chunking import chunk_text
from app.gemini_client import (
    embed_text, embed_image, caption_image, generate_answer, generate_answer_stream,
    generate_diagram, generate_structured, ocr_page_image,
    MAX_IMAGES_PER_DOCUMENT, MIN_IMAGE_DIMENSION,
)
from google.genai import errors as genai_errors
from psycopg2.extras import Json
from psycopg2 import Binary
from app.retrieval import (
    TOP_K, DISTANCE_THRESHOLD, DOC_MODE_CANDIDATE_K, DOC_MODE_CANDIDATE_MAX_DISTANCE,
    DOC_MODE_MAX_WINDOW, fetch_history, retrieve_chunks, fetch_chunk_window,
    build_sources, build_context_images,
)
from app.agent import build_graph, initial_state

_agent_graph = build_graph()  # compiled once at import time, reused across requests

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

def _paragraph_text_with_breaks(paragraph) -> str:
    """python-docx's paragraph.text concatenates runs but silently drops
    manual line breaks (Shift+Enter, <w:br/>) — which would destroy an ASCII
    tree/diagram typed as multiple lines inside one Word paragraph. Walk the
    paragraph's XML directly to preserve those as real newlines instead."""
    from docx.oxml.ns import qn
    pieces = []
    for run in paragraph.runs:
        for child in run._element:
            if child.tag == qn("w:t"):
                pieces.append(child.text or "")
            elif child.tag in (qn("w:br"), qn("w:cr")):
                pieces.append("\n")
            elif child.tag == qn("w:tab"):
                pieces.append("\t")
    return "".join(pieces)


def extract_docx_text(raw: bytes) -> str:
    """Extracts paragraph and table text from a .docx file.

    .docx is a ZIP archive of XML, not plain text — decoding its raw bytes
    as UTF-8 (the old fallback path) produces exactly the binary/XML garbage
    OpsMind was seeing and confidently answering questions about. This uses
    python-docx to walk the actual document structure instead.

    Note: .docx has no real page boundaries stored in the file — pagination
    is a rendering-time concept (depends on page size, margins, fonts), not
    something the XML records. Unlike PDFs, we can't tag chunks with an
    accurate page number here, so the whole document is treated as one
    logical page. That's an honest limitation, not a bug — don't try to
    fake page numbers for DOCX uploads.
    """
    document = docx.Document(io.BytesIO(raw))

    parts = [
        _paragraph_text_with_breaks(p)
        for p in document.paragraphs
        if p.text.strip()
    ]

    # SOPs/specs often carry key content (limits, thresholds, field names)
    # inside tables, not paragraphs — don't skip those.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)

    return "\n\n".join(parts)


def extract_pages(file: UploadFile, raw: bytes) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) tuples, 1-indexed.
    Plain text files (logs/SOPs) are treated as a single page (page 1) —
    there's no real pagination concept for those, but keeping the same
    return shape means upload_document doesn't need a separate code path."""
    filename = file.filename.lower()

    if filename.endswith(".doc") and not filename.endswith(".docx"):
        # Legacy binary Word format (pre-2007) — python-docx (and this whole
        # pipeline) only understands the newer .docx XML format. Reject this
        # explicitly instead of letting it silently fall through to the
        # plain-text branch and produce garbage the way .docx used to.
        raise HTTPException(
            400,
            "Legacy .doc files aren't supported — only .docx (Word 2007+). "
            "Re-save this file as .docx in Word and re-upload.",
        )

    if filename.endswith(".docx"):
        text = extract_docx_text(raw)
        return [(1, text)]

    if filename.endswith(".pdf"):
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


def extract_images(file: UploadFile, raw: bytes) -> list[tuple[int, bytes]]:
    """Returns (page_number, png_bytes) for embedded images worth indexing.
    Only PDFs can contain embedded images in this pipeline. Filters out tiny
    images (icons/bullets/logos — see MIN_IMAGE_DIMENSION) and hard-caps the
    total per document (MAX_IMAGES_PER_DOCUMENT) since each image costs two
    Gemini calls (caption + embed) — an uncapped image-heavy PDF could burn
    a meaningful chunk of a tight daily quota in one upload."""
    if not file.filename.lower().endswith(".pdf"):
        return []

    images = []
    pdf_doc = fitz.open(stream=raw, filetype="pdf")
    try:
        for page_index in range(len(pdf_doc)):
            if len(images) >= MAX_IMAGES_PER_DOCUMENT:
                break
            page = pdf_doc[page_index]
            for img in page.get_images(full=True):
                if len(images) >= MAX_IMAGES_PER_DOCUMENT:
                    break
                xref = img[0]
                try:
                    pix = fitz.Pixmap(pdf_doc, xref)
                    if pix.n - pix.alpha >= 4:  # CMYK or similar — convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < MIN_IMAGE_DIMENSION or pix.height < MIN_IMAGE_DIMENSION:
                        continue  # skip icons/logos/bullets — not worth a caption + embed call
                    images.append((page_index + 1, pix.tobytes("png")))  # normalized to PNG
                except Exception:
                    continue  # skip images that fail to extract rather than aborting the upload
    finally:
        pdf_doc.close()
    return images


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), doc_type: str = Form(...)):
    raw = await file.read()
    pages = extract_pages(file, raw)
    if not any(text.strip() for _, text in pages):
        raise HTTPException(400, "Couldn't extract any text from this file")

    images = extract_images(file, raw)  # capped + size-filtered inside extract_images

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
                page_text = page_text.replace(chr(0), "")
                if not page_text.strip():
                    continue
                # chunk each page independently — keeps page metadata accurate per chunk
                # instead of losing page boundaries by chunking the whole document at once
                for chunk in chunk_text(page_text):
                    embedding = embed_text(chunk)
                    cur.execute(
                        "INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, modality) "
                        "VALUES (%s, %s, %s, %s, %s, 'text')",
                        (document_id, chunk, embedding, chunk_index, Json({"page": page_num})),
                    )
                    chunk_index += 1

            images_indexed = 0
            for page_num, png_bytes in images:
                # two Gemini calls per image: one to caption it (readable content/fallback
                # text), one to embed it (into the SAME vector space as text, via
                # gemini-embedding-2 — see gemini_client.py)
                caption = caption_image(png_bytes)
                embedding = embed_image(png_bytes)
                cur.execute(
                    "INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, "
                    "modality, image_data) VALUES (%s, %s, %s, %s, %s, 'image', %s)",
                    (document_id, caption, embedding, chunk_index, Json({"page": page_num}), Binary(png_bytes)),
                )
                chunk_index += 1
                images_indexed += 1
        conn.commit()
    finally:
        conn.close()

    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunks_created": chunk_index,
        "images_indexed": images_indexed,
    }


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


def parse_command(question: str) -> tuple[str, str]:
    """Detects @doc / @art prefixes and returns (mode, cleaned_question).
    mode is one of: "normal", "doc", "art"."""
    stripped = question.strip()
    lower = stripped.lower()
    for prefix, mode in (("@doc", "doc"), ("@art", "art")):
        if lower.startswith(prefix):
            return mode, stripped[len(prefix):].strip()
    return "normal", stripped


# Matches common section-heading conventions: numbered headings like
# "9. WHOLE E-Z AUTO — 90–120 SECOND ANSWER" or short ALL-CAPS lines like
# "VENDOR TYPES & PRIORITY LOGIC". Generic enough to work across differently
# structured documents — not specific to any one file's formatting.
_HEADING_PATTERN = re.compile(r"^\s{0,3}(?:\d{1,2}\.\s+[A-Z]|[A-Z][A-Z \-\u2014]{4,}$)", re.MULTILINE)


def _extract_section_text(combined: str, anchor_text: str) -> str:
    """Given a wide band of concatenated chunk text and the specific chunk
    that was selected as the best match, trims to just the section that
    chunk belongs to — starting AFTER its heading line, up to (but not
    including) the next heading.

    This exists because a fixed-size chunk window either truncates long
    sections or bleeds into unrelated neighboring ones; the actual section
    boundary is the thing that generalizes correctly across documents, not
    a guessed window size. Falls back to the full window if no heading
    pattern is detected (e.g. documents that don't use numbered/ALL-CAPS
    section headers) — better to return more than the ideal slice than to
    return nothing for an unstructured document.

    Deliberately excludes the heading line itself from the returned answer —
    @doc mode should return the actual answer, not the section title as if
    it were part of the answer (e.g. skip "9. WHOLE E-Z AUTO — 90-120 SECOND
    ANSWER" and start directly with the real paragraph content that follows).
    """
    anchor_pos = combined.find(anchor_text[:80])
    anchor_pos = max(anchor_pos, 0)

    heading_matches = list(_HEADING_PATTERN.finditer(combined))
    at_or_after_anchor = [m for m in heading_matches if m.start() >= anchor_pos]

    if not at_or_after_anchor:
        return combined.strip()  # no heading convention detected — return the full window

    section_heading = at_or_after_anchor[0]
    # The heading regex only matches the OPENING of a heading line (e.g. just
    # "9. W" of "9. WHOLE E-Z AUTO..."), not the full line — find where that
    # whole line actually ends so we skip the complete heading, not a fragment.
    line_end = combined.find("\n", section_heading.end())
    content_start = line_end + 1 if line_end != -1 else section_heading.end()

    later_headings = at_or_after_anchor[1:]
    section_end = later_headings[0].start() if later_headings else len(combined)
    return combined[content_start:section_end].strip()


def resolve_doc_mode(cur, question: str, document_ids):
    """@doc mode: retrieves a WIDE candidate pool (not just the single top
    match), then uses one lightweight LLM call whose ONLY job is to pick
    which candidate genuinely answers the question — it never generates or
    rewrites the answer itself. Once selected, the answer is reconstructed
    from the actual stored chunk text (plus its immediate neighbors, to
    recover a section our chunker may have split across a boundary) and
    returned VERBATIM. No generation step touches the answer text, so
    wording/paragraph structure/formatting is guaranteed unchanged from the
    source document.

    Why not just take the single closest chunk by raw cosine distance (the
    old approach)? Loosely-phrased, holistic questions ("brief me about the
    whole project") are often NOT closest-by-embedding to the one section
    that actually answers them best — a document full of narrow Q&A chunks
    competes with a holistic summary section on shared vocabulary alone.
    Cosine similarity doesn't know "this question wants the summary
    specifically." An LLM comparing the question against several real
    candidates does understand that, which is a judgment call, not a
    distance calculation — hence the added selection call.

    Returns (answer, sources). answer is an explicit refusal message if no
    candidate genuinely answers the question — @doc mode never fabricates.
    """
    question_embedding = embed_text(question)
    candidates = retrieve_chunks(cur, question_embedding, document_ids, top_k=DOC_MODE_CANDIDATE_K)
    candidates = [c for c in candidates if c["distance"] <= DOC_MODE_CANDIDATE_MAX_DISTANCE]

    if not candidates:
        return (
            "No matching answer found in the uploaded document(s) for this. "
            "Try rephrasing, or ask without @doc for a synthesized answer instead.",
            [],
        )

    listing = "\n\n".join(
        f"[{i}] (from {c['filename']}): {c['content'][:300]}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"User's question: {question}\n\n"
        f"Candidate document excerpts:\n{listing}\n\n"
        "Which excerpt, if any, most directly and completely answers the user's question "
        "AS WRITTEN IN THE SOURCE — not just a related topic? Prefer a holistic/summary "
        "excerpt over a narrower one if the question is asking broadly (e.g. 'brief me "
        'about the whole project"). Respond with ONLY JSON: {"best_index": <int or null>}. '
        "Use null if none of the excerpts genuinely answer this — do not pick the "
        "closest-sounding one if it doesn't actually address what's being asked."
    )
    system_instruction = (
        "You are a precise document-matching assistant. You SELECT which excerpt best "
        "answers a question — you do not answer the question yourself, and you do not "
        "invent, summarize, or supplement anything. Respond with strict JSON only."
    )
    try:
        result = generate_structured(prompt, system_instruction)
        best_index = result.get("best_index")
    except Exception as e:
        # selection call failed — fall back to raw top-1 rather than fail the
        # whole request; better than nothing, even if less reliable
        print(f"[chat/@doc] Selection call failed, falling back to top match: {e}")
        best_index = 0

    if best_index is None or not isinstance(best_index, int) or not (0 <= best_index < len(candidates)):
        return (
            "No matching answer found in the uploaded document(s) for this. "
            "Try rephrasing, or ask without @doc for a synthesized answer instead.",
            [],
        )

    best = candidates[best_index]
    if best["modality"] == "image":
        # an image's caption IS its closest textual "answer" — no windowing
        # concept applies to a single image the way it does to text sections
        answer = best["content"]
    else:
        window_rows = fetch_chunk_window(cur, best["document_id"], best["chunk_index"])
        combined = "\n\n".join(r["content"] for r in window_rows) if window_rows else best["content"]
        answer = _extract_section_text(combined, best["content"])

    # Deliberately NOT calling build_sources here. The answer text above IS the
    # full source content, verbatim — an expandable "sourced:" chip that reveals
    # a snippet of the same content the user is already reading would just be a
    # duplicate, not genuine attribution. @doc's answer speaks for itself.
    return answer, []


@app.post("/chat")
def chat(req: ChatRequest):
    mode, question = parse_command(req.question)

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
            history = fetch_history(cur, conversation_id)

            if mode == "doc":
                # @doc: retrieval + verbatim reconstruction only, no generation call at all
                answer, sources = resolve_doc_mode(cur, question, req.document_ids)

            elif mode == "art":
                # @art: same topical-relevance gate as normal chat, but generation
                # produces a grounded Mermaid diagram instead of prose
                question_embedding = embed_text(question)
                retrieved = retrieve_chunks(cur, question_embedding, req.document_ids)
                if not retrieved or retrieved[0]["distance"] > DISTANCE_THRESHOLD:
                    answer = "Couldn't find enough relevant document content to build a diagram for this."
                    sources = []
                else:
                    context_chunks = [f"[{r['filename']}] {r['content']}" for r in retrieved]
                    try:
                        answer = generate_diagram(question, context_chunks)
                        sources = build_sources(retrieved)
                    except genai_errors.ClientError as e:
                        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                            print(f"[chat/@art] All models hit quota/limits: {e}")
                            answer = (
                                "This app has hit today's free-tier request limit across every "
                                "available Gemini model. It'll reset within a day — try again later."
                            )
                        else:
                            print(f"[chat/@art] Gemini ClientError: {e}")
                            answer = "Something went wrong talking to Gemini. Check the server logs for details."
                        sources = []
                    except genai_errors.ServerError as e:
                        print(f"[chat/@art] All models returned ServerError: {e}")
                        answer = "Gemini's servers are temporarily overloaded. Please try asking again in a moment."
                        sources = []

            else:
                # 3. embed the question and retrieve the most relevant chunks,
                #    optionally scoped to specific documents
                question_embedding = embed_text(question)
                retrieved = retrieve_chunks(cur, question_embedding, req.document_ids)

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
                    context_images = build_context_images(retrieved)
                    try:
                        answer = generate_answer(question, context_chunks, history, context_images)
                        sources = build_sources(retrieved)
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
    mode, question = parse_command(req.question)
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

                history = fetch_history(cur, conversation_id)

                if mode == "doc":
                    # pure DB reconstruction — no LLM call, so no real streaming benefit;
                    # deliver as one chunk event so the frontend's handling stays uniform
                    answer, sources = resolve_doc_mode(cur, question, req.document_ids)
                    yield _sse({"type": "chunk", "text": answer})

                elif mode == "art":
                    # a diagram is only valid once complete — token-streaming it would
                    # render broken partial Mermaid syntax mid-flight, so this is
                    # deliberately delivered as one complete chunk, not streamed
                    question_embedding = embed_text(question)
                    retrieved = retrieve_chunks(cur, question_embedding, req.document_ids)
                    if not retrieved or retrieved[0]["distance"] > DISTANCE_THRESHOLD:
                        answer = "Couldn't find enough relevant document content to build a diagram for this."
                        sources = []
                    else:
                        context_chunks = [f"[{r['filename']}] {r['content']}" for r in retrieved]
                        try:
                            answer = generate_diagram(question, context_chunks)
                            sources = build_sources(retrieved)
                        except genai_errors.ClientError as e:
                            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                                print(f"[chat/stream/@art] All models hit quota/limits: {e}")
                                answer = (
                                    "This app has hit today's free-tier request limit across every "
                                    "available Gemini model. It'll reset within a day — try again later."
                                )
                            else:
                                print(f"[chat/stream/@art] Gemini ClientError: {e}")
                                answer = "Something went wrong talking to Gemini. Check the server logs for details."
                            sources = []
                        except genai_errors.ServerError as e:
                            print(f"[chat/stream/@art] All models returned ServerError: {e}")
                            answer = "Gemini's servers are temporarily overloaded. Please try asking again in a moment."
                            sources = []
                    yield _sse({"type": "chunk", "text": answer})

                else:
                    # 2. history + retrieval — identical to the non-streaming /chat
                    question_embedding = embed_text(question)
                    retrieved = retrieve_chunks(cur, question_embedding, req.document_ids)

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
                        context_images = build_context_images(retrieved)
                        sources = build_sources(retrieved)
                        answer_parts = []
                        try:
                            for piece in generate_answer_stream(question, context_chunks, history, context_images):
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


# ---------- Agentic chat ----------

@app.post("/chat/agent")
def chat_agent(req: ChatRequest):
    """
    Runs the LangGraph agent (app/agent.py) instead of the linear /chat pipeline.
    Deliberately a SEPARATE endpoint rather than replacing /chat — the existing
    RAG pipeline is working and battle-tested; this is additive, not a rewrite.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            conversation_id = req.conversation_id
            if conversation_id is None:
                cur.execute(
                    "INSERT INTO conversations (title) VALUES (%s) RETURNING id",
                    (req.question[:60],),
                )
                conversation_id = cur.fetchone()["id"]
            history = fetch_history(cur, conversation_id)
        conn.commit()
    finally:
        conn.close()

    state = initial_state(req.question, history, req.document_ids)

    try:
        final_state = _agent_graph.invoke(state)
    except genai_errors.ClientError as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            print(f"[chat/agent] All models hit quota/limits: {e}")
            answer = (
                "This app has hit today's free-tier request limit across every "
                "available Gemini model. It'll reset within a day — try again later."
            )
        else:
            print(f"[chat/agent] Gemini ClientError: {e}")
            answer = "Something went wrong talking to Gemini. Check the server logs for details."
        final_state = {"answer": answer, "sources": [], "intent": "direct", "investigation_steps": []}
    except genai_errors.ServerError as e:
        print(f"[chat/agent] All models returned ServerError: {e}")
        final_state = {
            "answer": "Gemini's servers are temporarily overloaded. Please try asking again in a moment.",
            "sources": [], "intent": "direct", "investigation_steps": [],
        }

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
                (conversation_id, req.question),
            )
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'assistant', %s)",
                (conversation_id, final_state["answer"]),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "conversation_id": conversation_id,
        "answer": final_state["answer"],
        "sources": final_state.get("sources", []),
        "intent": final_state.get("intent"),
        "investigation_steps": final_state.get("investigation_steps", []),
    }

