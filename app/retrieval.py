import base64

TOP_K = 5
# Cosine distance from pgvector's <=> operator: 0 = identical, 2 = opposite.
# Above this, the "best" match is too dissimilar to trust — treat as no relevant context.
# Tune this against your own data; it's a heuristic, not a physical constant.
DISTANCE_THRESHOLD = 0.6

# @doc mode: a wider candidate pool than normal chat, because a loosely-phrased
# question ("brief me about the whole project") is often NOT closest-by-raw-cosine
# to the actual best section — a document full of narrow Q&A chunks competes with
# the one holistic summary section on shared vocabulary alone. Relevance judgment
# for @doc happens via LLM selection (see resolve_doc_mode in main.py), not a
# single strict distance cutoff — so this is a loose sanity filter, not a
# precision gate the way DISTANCE_THRESHOLD is for normal chat.
DOC_MODE_CANDIDATE_K = 10
DOC_MODE_CANDIDATE_MAX_DISTANCE = 0.9

# @doc mode: how many adjacent chunks to fetch (by chunk_index) around a selected
# match, BEFORE trimming to the actual section boundaries (see _extract_section_text
# in main.py). Fetched wide on purpose — precision comes from heading-based trimming
# afterward, not from guessing the right window size, since a fixed narrow window
# either truncates long sections or bleeds into unrelated neighboring ones.
DOC_MODE_MAX_WINDOW = 4


def fetch_history(cur, conversation_id):
    cur.execute(
        "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at",
        (conversation_id,),
    )
    return cur.fetchall()


def retrieve_chunks(cur, question_embedding, document_ids, top_k=TOP_K):
    query = (
        "SELECT c.content, d.filename, d.id AS document_id, c.metadata, "
        "c.modality, c.image_data, c.chunk_index, "
        "(c.embedding <=> %s::vector) AS distance "
        "FROM chunks c JOIN documents d ON d.id = c.document_id"
    )
    params = [question_embedding]
    if document_ids:
        query += " WHERE c.document_id = ANY(%s)"
        params.append(document_ids)
    query += " ORDER BY distance LIMIT %s"
    params.append(top_k)
    cur.execute(query, params)
    return cur.fetchall()


def fetch_chunk_window(cur, document_id, center_index, window=DOC_MODE_MAX_WINDOW):
    """Pulls a wide band of a selected chunk's TEXT neighbors (same document,
    consecutive chunk_index). Deliberately over-fetches — the caller trims
    this down to the actual section boundaries afterward."""
    cur.execute(
        "SELECT content, chunk_index FROM chunks "
        "WHERE document_id = %s AND chunk_index BETWEEN %s AND %s AND modality = 'text' "
        "ORDER BY chunk_index",
        (document_id, center_index - window, center_index + window),
    )
    return cur.fetchall()


def build_sources(retrieved):
    sources = []
    for r in retrieved:
        entry = {
            "filename": r["filename"],
            "snippet": r["content"][:400],
            "page": (r["metadata"] or {}).get("page"),
            "modality": r["modality"],
        }
        if r["modality"] == "image" and r["image_data"]:
            # small enough for a UI thumbnail; not used for retrieval, display only
            entry["image_base64"] = base64.b64encode(bytes(r["image_data"])).decode("ascii")
        sources.append(entry)
    return sources


def build_context_images(retrieved):
    """Raw bytes for every retrieved image chunk, in the shape generate_answer
    expects — so Gemini actually SEES the image during generation, not just
    a caption of it."""
    return [
        {"data": bytes(r["image_data"]), "mime_type": "image/png"}
        for r in retrieved
        if r["modality"] == "image" and r["image_data"]
    ]
