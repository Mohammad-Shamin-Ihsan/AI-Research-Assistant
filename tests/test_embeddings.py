import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_embeddings import generate_embeddings, generate_single_embedding
import math


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def test_embedding_dimensions():
    """Test that embeddings have correct dimensions."""

    print("Test 1: Embedding dimensions...")

    text = "This is a test sentence."
    embedding = generate_single_embedding(text)

    # nomic-embed-text produces 768-dimensional embeddings
    expected_dim = 768
    actual_dim = len(embedding)

    assert actual_dim == expected_dim, f"Expected {expected_dim} dimensions, got {actual_dim}"
    print(f"✓ Embedding has correct dimensions: {actual_dim}")


def test_multiple_embeddings():
    """Test generating embeddings for multiple chunks."""

    print("\nTest 2: Multiple embeddings...")

    chunks = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "The weather is sunny today."
    ]

    embeddings = generate_embeddings(chunks)

    assert len(embeddings) == len(chunks), f"Expected {len(chunks)} embeddings, got {len(embeddings)}"
    print(f"✓ Generated {len(embeddings)} embeddings for {len(chunks)} chunks")


def test_semantic_similarity():
    """Test that similar texts have higher similarity scores."""

    print("\nTest 3: Semantic similarity...")

    # Similar texts
    text1 = "Machine learning is used in artificial intelligence."
    text2 = "AI systems often use machine learning algorithms."

    # Unrelated text
    text3 = "The pizza was delicious for dinner."

    emb1 = generate_single_embedding(text1)
    emb2 = generate_single_embedding(text2)
    emb3 = generate_single_embedding(text3)

    # Calculate similarities
    similarity_related = cosine_similarity(emb1, emb2)
    similarity_unrelated = cosine_similarity(emb1, emb3)

    print(f"  Similarity (related texts): {similarity_related:.4f}")
    print(f"  Similarity (unrelated texts): {similarity_unrelated:.4f}")

    # Related texts should be more similar than unrelated texts
    assert similarity_related > similarity_unrelated, \
        "Related texts should have higher similarity than unrelated texts"

    print("✓ Similar texts have higher similarity scores")


def test_empty_input():
    """Test handling of empty input."""

    print("\nTest 4: Empty input handling...")

    # Empty list should return empty list
    embeddings = generate_embeddings([])
    assert embeddings == [], "Empty input should return empty list"

    # Empty string should raise error
    try:
        generate_single_embedding("")
        assert False, "Empty string should raise ValueError"
    except ValueError:
        print("✓ Empty inputs handled correctly")


def test_deterministic_output():
    """Test that same input produces same embedding."""

    print("\nTest 5: Deterministic output...")

    text = "Consistency test for embeddings."

    emb1 = generate_single_embedding(text)
    emb2 = generate_single_embedding(text)

    # Check if embeddings are identical
    similarity = cosine_similarity(emb1, emb2)

    # Should be exactly 1.0 (or very close due to floating point)
    assert similarity > 0.9999, f"Same input should produce same output, got similarity {similarity}"
    print(f"✓ Same input produces consistent output (similarity: {similarity:.6f})")


if __name__ == "__main__":

    print("="*60)
    print("Running Embedding Tests")
    print("="*60 + "\n")

    try:
        test_embedding_dimensions()
        test_multiple_embeddings()
        test_semantic_similarity()
        test_empty_input()
        test_deterministic_output()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error running tests: {e}")
        sys.exit(1)
