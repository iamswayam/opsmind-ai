import os
import time
import json
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API Key not found. Please set GEMINI_API_KEY environment variable.")

# Default client uses the v1beta surface, which supports systemInstruction.
# DO NOT pin http_options to api_version "v1" — that older surface doesn't
# recognize systemInstruction/responseMimeType/responseSchema and will 400.
client = genai.Client(api_key=api_key)

# Verify these against `client.models.list()` or ai.google.dev if you hit another 404 —
# model names on the free tier get renamed/deprecated faster than this comment will stay accurate.
#
# gemini-embedding-001 is TEXT-ONLY. gemini-embedding-2 is Google's multimodal embedding
# model (GA as of April 2026) — it embeds text AND images into the SAME vector space,
# which is what makes real cross-modal retrieval possible (a text question's embedding
# can be genuinely close to a relevant image's embedding, not just to other text chunks).
# Both text and images are embedded with this ONE model so they share that space.
EMBED_MODEL = "gemini-embedding-2"

# Safety limits for image processing — each image costs TWO Gemini calls (caption +
# embed), on top of whatever quota text chunking already uses. Given how tight this
# project's free-tier daily quotas have turned out to be in practice, an uncapped PDF
# with 50 embedded logos/icons could burn the entire daily budget on one upload.
MAX_IMAGES_PER_DOCUMENT = 10
MIN_IMAGE_DIMENSION = 100  # px — skips tiny icons/bullets/logos not worth captioning

# Ordered by actual daily quota headroom on this project's AI Studio dashboard
# (checked directly — don't trust blog posts on this, quotas vary by project/date).
# Highest RPD first. When one is exhausted or blocked, the next is tried automatically.
CHAT_MODEL_FALLBACK_CHAIN = [
    "gemini-3.1-flash-lite",   # 500 RPD — by far the most headroom of any text model here
    "gemini-2.5-flash-lite",   # 20 RPD
    "gemini-3-flash",          # 20 RPD
    "gemini-2.5-flash",        # 20 RPD — may 404 as "no longer available to new users" on some projects
    "gemini-3.5-flash",        # 20 RPD — was already exhausted as of this session
]


def _generate_with_fallback(contents, config=None):
    """Shared call path for anything that needs client.models.generate_content.
    Tries each model in CHAT_MODEL_FALLBACK_CHAIN in order:
      - on a transient 5xx ServerError, retries the SAME model with backoff first
      - on a 429 quota error or 404 (deprecated/unavailable) ClientError, moves to
        the NEXT model in the chain instead of retrying — those won't clear in seconds
      - on any other ClientError (e.g. malformed request), raises immediately —
        that's a real bug, not something falling back to another model would fix
    Raises the last error only if every model in the chain fails.
    """
    last_error = None
    for model_name in CHAT_MODEL_FALLBACK_CHAIN:
        for attempt in range(3):
            try:
                kwargs = {"model": model_name, "contents": contents}
                if config is not None:
                    kwargs["config"] = config
                response = client.models.generate_content(**kwargs)
                return response
            except genai_errors.ServerError as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, then 2s — retry same model
                    continue
                break  # exhausted retries for this model, fall through to next model
            except genai_errors.ClientError as e:
                last_error = e
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "NOT_FOUND" in msg or "404" in msg:
                    break  # quota hit or model unavailable — try the next model in the chain
                raise  # some other client error (bad request, auth, etc.) — don't mask real bugs
    raise last_error


def generate_structured(prompt: str, system_instruction: str) -> dict:
    """Requests a strict JSON response from Gemini and parses it into a dict.
    Used by the agent's triage and investigation nodes, which need structured
    decisions (intent, next_action, etc.) rather than free-text prose.

    Goes through the same _generate_with_fallback model chain as everything
    else — a structured call is just as subject to quota/availability issues
    as a normal generation call, so it needs the same resilience.

    Raises ValueError if the response isn't valid JSON. Deliberately doesn't
    swallow this — a node getting malformed JSON back is a real problem the
    caller needs to know about, not something to silently paper over with a
    guessed default.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )
    response = _generate_with_fallback(contents=prompt, config=config)
    text = (response.text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Expected JSON but got: {text[:200]}") from e


def embed_text(text: str) -> list[float]:
    """Turn a chunk of text (or a user question) into a 768-dim vector.
    gemini-embedding-2 returns higher-dim vectors by default, so we truncate
    to 768 to match the pgvector column (see chunks.embedding VECTOR(768))."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return result.embeddings[0].values


def embed_image(image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
    """Embeds an image into the SAME 768-dim vector space as embed_text, via
    gemini-embedding-2. This is what enables true cross-modal retrieval — a
    text question can be compared directly against an image's embedding.

    Multimodal embedding is a newer capability than plain text embedding —
    if this call shape errors, check ai.google.dev for the current expected
    input format before assuming the model itself is unavailable."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return result.embeddings[0].values


def caption_image(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Generates a short, concrete caption for an extracted image — becomes
    the chunk's `content` text (used for display and as fallback readable
    context), separate from the image's own embedding."""
    response = _generate_with_fallback(
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Describe this image in one or two sentences: what it shows, and any "
            "key labels, values, or diagram structure visible. Be concrete and "
            "specific rather than generic — mention actual text/numbers you can read.",
        ]
    )
    return (response.text or "").strip()


def ocr_page_image(image_bytes: bytes) -> str:
    """Transcribe a single rendered PDF page image using Gemini's vision input.
    Used as a fallback when pypdf finds no extractable text layer (i.e. the
    PDF is scanned/image-based rather than text-based)."""
    response = _generate_with_fallback(
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            "Transcribe all readable text from this document page, in reading order. "
            "Return only the transcribed text, no commentary.",
        ]
    )
    return response.text or ""


def generate_diagram(question: str, context_chunks: list[str]) -> str:
    """@art mode: generates a Mermaid diagram definition grounded in retrieved
    document content, wrapped in a fenced code block the frontend detects and
    renders as an actual SVG diagram (see renderMermaidBlocks in index.html).

    Delivered as ONE complete response, never token-streamed — a Mermaid
    diagram is only valid once its syntax is complete, so streaming it
    character-by-character would just render broken partial diagrams mid-flight.
    """
    context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context found)"
    system_instruction = (
        "You are a technical diagramming assistant. Given retrieved document context, "
        "produce ONE Mermaid diagram (use 'flowchart TD' syntax) that visually represents "
        "the architecture, process flow, or mental model being asked about.\n\n"
        "LAYOUT RULES — these matter as much as content accuracy. A diagram that sprawls "
        "wide is a failure even if every fact in it is correct:\n"
        "- Keep the diagram COMPACT and mostly vertical. Never place more than 3 sibling "
        "nodes side-by-side at the same rank. If more than 3 related components exist at "
        "one conceptual level, group them inside a subgraph instead of laying them all out "
        "in one wide row.\n"
        "- Use subgraphs to cluster logically related components (e.g. all external "
        "services together, all steps of one pipeline together) rather than scattering "
        "them as separate top-level nodes connected by long, crossing arrows.\n"
        "- Prefer short, direct connections between adjacent concepts. Do not route an "
        "arrow across the whole diagram when the two connected nodes could instead be "
        "placed near each other.\n"
        "- Keep total node count to roughly 8-14. This is a conceptual overview, not an "
        "exhaustive inventory — omit minor details that would only add width without "
        "adding understanding.\n"
        "- Keep node labels to 2-4 words. Long labels widen the diagram unnecessarily.\n\n"
        "Base the diagram ONLY on what's actually described in the provided context — do "
        "not invent components, steps, or relationships that aren't supported by it. If the "
        "context describes a tree/hierarchy, use subgraphs or nested nodes to mirror that "
        "structure rather than flattening it.\n\n"
        "Output ONLY valid Mermaid syntax — no explanation before or after, no markdown "
        "code fences (those are added separately)."
    )
    prompt = f"Context:\n{context_block}\n\nCreate a diagram for: {question}"

    response = _generate_with_fallback(
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    mermaid_code = (response.text or "").strip()
    # strip accidental fences if the model added them despite being told not to
    for fence in ("```mermaid", "```"):
        if mermaid_code.startswith(fence):
            mermaid_code = mermaid_code[len(fence):].strip()
    if mermaid_code.endswith("```"):
        mermaid_code = mermaid_code[:-3].strip()
    return f"```mermaid\n{mermaid_code}\n```"


def _build_prompt(question: str, context_chunks: list[str]):
    """Shared prompt + system instruction builder for both generate_answer
    and generate_answer_stream, so the two paths can't silently drift apart."""
    context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context found)"

    system_instruction = (
        "You are OpsMind, an internal assistant for a support/operations team. "
        "Answer using ONLY the provided context (SOPs, logs, incident reports, API docs). "
        "If the context doesn't contain the answer, say so explicitly instead of guessing. "
        "Do NOT preface your answer with phrases like 'Based on the document X' or "
        "'According to [filename]' — the source is already shown separately in the UI below "
        "your answer, so just answer the question directly, as if you already know it. "
        "Format for a chat bubble: use bold for emphasis and short bullet lists where helpful, "
        "but avoid headers (###) and deep nesting — keep it scannable, not document-styled."
    )

    prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
    return prompt, system_instruction


def generate_answer(question: str, context_chunks: list[str], history: list[dict],
                     context_images: list[dict] | None = None) -> str:
    """
    context_chunks: retrieved SOP/log/incident text snippets (image chunks' captions
                     are already included here as text — see main.py/agent.py)
    context_images: retrieved image chunks' RAW BYTES, so Gemini can actually see the
                     image during generation, not just read a caption of it.
                     [{"data": bytes, "mime_type": "image/png"}, ...]
    history: prior turns in this conversation, [{"role": "user"/"assistant", "content": "..."}]
    """
    prompt, system_instruction = _build_prompt(question, context_chunks)
    contents = [prompt] + [
        types.Part.from_bytes(data=img["data"], mime_type=img.get("mime_type", "image/png"))
        for img in (context_images or [])
    ]

    response = _generate_with_fallback(
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text


def generate_answer_stream(question: str, context_chunks: list[str], history: list[dict],
                            context_images: list[dict] | None = None):
    """
    Yields text deltas as they arrive from Gemini, trying each model in
    CHAT_MODEL_FALLBACK_CHAIN in order. See generate_answer for context_images shape.

    Trade-off worth knowing: fallback here only works if a model fails BEFORE
    yielding any text (e.g. an immediate 429/404 on connect). If a model starts
    streaming and then fails mid-response, we can't cleanly retry on another
    model without either duplicating already-sent text or discarding it, so
    that case just stops the stream where it is. This is rarer in practice —
    most failures happen on the initial request, not mid-stream — but it's a
    real limitation, not an oversight.
    """
    prompt, system_instruction = _build_prompt(question, context_chunks)
    contents = [prompt] + [
        types.Part.from_bytes(data=img["data"], mime_type=img.get("mime_type", "image/png"))
        for img in (context_images or [])
    ]
    config = types.GenerateContentConfig(system_instruction=system_instruction)

    last_error = None
    for model_name in CHAT_MODEL_FALLBACK_CHAIN:
        try:
            stream = client.models.generate_content_stream(
                model=model_name, contents=contents, config=config
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return  # this model completed the stream successfully
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            last_error = e
            msg = str(e)
            is_retryable = (
                isinstance(e, genai_errors.ServerError)
                or "RESOURCE_EXHAUSTED" in msg or "429" in msg
                or "NOT_FOUND" in msg or "404" in msg
            )
            if not is_retryable:
                raise  # real bug (bad request, auth) — don't mask it by trying other models
            continue  # try the next model in the chain
    raise last_error
