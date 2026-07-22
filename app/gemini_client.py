import os
import time
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
EMBED_MODEL = "gemini-embedding-001"   # NOT "embedding-001" — that name 404s under v1

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


def embed_text(text: str) -> list[float]:
    """Turn a chunk of text (or a user question) into a 768-dim vector.
    gemini-embedding-001 returns 3072 dims by default, so we truncate to 768
    to match the pgvector column (see chunks.embedding VECTOR(768) in db/init.sql)."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return result.embeddings[0].values


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


def generate_answer(question: str, context_chunks: list[str], history: list[dict]) -> str:
    """
    context_chunks: the retrieved SOP/log/incident snippets from pgvector
    history: prior turns in this conversation, [{"role": "user"/"assistant", "content": "..."}]
    """
    prompt, system_instruction = _build_prompt(question, context_chunks)

    response = _generate_with_fallback(
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text


def generate_answer_stream(question: str, context_chunks: list[str], history: list[dict]):
    """
    Yields text deltas as they arrive from Gemini, trying each model in
    CHAT_MODEL_FALLBACK_CHAIN in order.

    Trade-off worth knowing: fallback here only works if a model fails BEFORE
    yielding any text (e.g. an immediate 429/404 on connect). If a model starts
    streaming and then fails mid-response, we can't cleanly retry on another
    model without either duplicating already-sent text or discarding it, so
    that case just stops the stream where it is. This is rarer in practice —
    most failures happen on the initial request, not mid-stream — but it's a
    real limitation, not an oversight.
    """
    prompt, system_instruction = _build_prompt(question, context_chunks)
    config = types.GenerateContentConfig(system_instruction=system_instruction)

    last_error = None
    for model_name in CHAT_MODEL_FALLBACK_CHAIN:
        try:
            stream = client.models.generate_content_stream(
                model=model_name, contents=prompt, config=config
            )
            yielded_anything = False
            for chunk in stream:
                if chunk.text:
                    yielded_anything = True
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