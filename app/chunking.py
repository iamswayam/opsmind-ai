def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Naive fixed-size chunking with overlap. This is intentionally simple —
    good enough to get RAG working end to end on Day 3-4.

    Once the pipeline works, come back and try:
      - splitting on paragraph/section boundaries instead of raw character count
      - variable chunk size for logs (line-based) vs SOPs (paragraph-based)
    That comparison is a great interview talking point: "naive chunking worked,
    but I found X% better retrieval quality after switching to Y strategy."
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step back by `overlap` so context isn't lost at chunk edges
    return [c.strip() for c in chunks if c.strip()]
