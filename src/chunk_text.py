def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200
) -> list[str]:
    """
    Split text into smaller chunks with overlapping content.

    Args:
        text: The extracted document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between adjacent chunks.

    Returns:
        A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Remove unnecessary whitespace around the document.
    text = text.strip()

    if not text:
        return []

    chunks = []

    # Begin at the first character.
    start = 0

    while start < len(text):

        # Calculate the end position of the current chunk.
        end = start + chunk_size

        # Extract the current piece of text.
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop when we have reached the end of the document.
        if end >= len(text):
            break

        # Start the next chunk before the previous one ended.
        # This creates an overlap between adjacent chunks.
        start = end - overlap

    return chunks


if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is increasingly used in medical imaging.
    Deep learning models can identify patterns in chest X-ray images.
    However, model performance depends on data quality and diversity.
    """

    chunks = chunk_text(
        sample_text,
        chunk_size=100,
        overlap=20
    )

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)