def _fixed_size_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback for text with no usable paragraph breaks (e.g. dense logs) —
    the original naive strategy, kept as a safety net rather than the default."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Paragraph-aware chunking: groups whole paragraphs together up to ~chunk_size,
    instead of cutting mid-sentence at a fixed character count. This matters for
    retrieval quality — a chunk that ends mid-sentence loses context that a
    paragraph boundary preserves.

    Falls back to _fixed_size_split for any single paragraph that's larger than
    chunk_size on its own (common in dense logs with no blank-line breaks) —
    there's no clean boundary to respect in that case, so naive splitting is
    the honest fallback rather than the default strategy.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # adding this paragraph would overflow the current chunk — close it out first
        if current:
            chunks.append(current)
            current = ""

        if len(para) > chunk_size:
            # this single paragraph is too big to fit in one chunk on its own
            chunks.extend(_fixed_size_split(para, chunk_size, overlap))
        else:
            current = para

    if current:
        chunks.append(current)

    return chunks