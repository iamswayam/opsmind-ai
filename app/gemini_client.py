import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"   # check ai.google.dev for the current embedding model name
CHAT_MODEL = "gemini-2.5-flash"        # fast + free-tier friendly


def embed_text(text: str) -> list[float]:
    """Turn a chunk of text (or a user question) into a 768-dim vector."""
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def generate_answer(question: str, context_chunks: list[str], history: list[dict]) -> str:
    """
    Non-streaming first pass — get this working before adding streaming.
    context_chunks: the retrieved SOP/log/incident snippets from pgvector
    history: prior turns in this conversation, [{"role": "user"/"assistant", "content": "..."}]
    """
    context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context found)"

    system_instruction = (
        "You are OpsMind, an internal assistant for a support/operations team. "
        "Answer using ONLY the provided context (SOPs, logs, incident reports, API docs). "
        "If the context doesn't contain the answer, say so explicitly instead of guessing. "
        "When relevant, cite which document the information came from."
    )

    # TODO (Day 5): fold `history` into the prompt or into a multi-turn chat session
    # so follow-up questions like "what about the retry logic?" resolve correctly.
    prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config={"system_instruction": system_instruction},
    )
    return response.text


def generate_answer_stream(question: str, context_chunks: list[str], history: list[dict]):
    """
    TODO (Day 5-6): implement this using client.models.generate_content_stream(...)
    and yield chunks so main.py can return a StreamingResponse.
    Get generate_answer() working and tested first — streaming is a wrapper
    around the same prompt, not a different concept.
    """
    raise NotImplementedError("Build this after generate_answer() is working end to end")
