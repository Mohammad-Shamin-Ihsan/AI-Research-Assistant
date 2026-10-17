import ollama


def generate_embeddings(chunks: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks using Ollama.

    Args:
        chunks: List of text chunks to embed.
        model: Name of the Ollama embedding model to use.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """

    if not chunks:
        return []

    embeddings = []

    for chunk in chunks:
        try:
            # Generate embedding for this chunk
            response = ollama.embeddings(
                model=model,
                prompt=chunk
            )

            # Extract the embedding vector
            embedding = response['embedding']
            embeddings.append(embedding)

        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    return embeddings


def generate_single_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    """
    Generate an embedding for a single text string.

    Args:
        text: Text to embed.
        model: Name of the Ollama embedding model to use.

    Returns:
        An embedding vector as a list of floats.
    """

    if not text:
        raise ValueError("Text cannot be empty")

    try:
        response = ollama.embeddings(
            model=model,
            prompt=text
        )
        return response['embedding']

    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}")


if __name__ == "__main__":

    # Test with sample chunks
    sample_chunks = [
        "Artificial intelligence is transforming healthcare.",
        "Machine learning models can detect diseases early.",
        "The sky is blue because of Rayleigh scattering."
    ]

    print("Generating embeddings for sample chunks...\n")

    embeddings = generate_embeddings(sample_chunks)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Test single embedding
    print("\nGenerating single embedding...")
    single_emb = generate_single_embedding("This is a test sentence.")
    print(f"Single embedding dimension: {len(single_emb)}")

    # Show first few values of first embedding
    print(f"\nFirst 10 values of first embedding:")
    print(embeddings[0][:10])
