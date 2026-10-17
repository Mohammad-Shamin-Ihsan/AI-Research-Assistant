import sys
from pathlib import Path
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import (
    retrieve_relevant_chunks,
    format_results,
    build_context,
    build_prompt_with_context,
    rerank_by_keywords
)
from vector_store import (
    initialize_db,
    get_collection,
    add_documents,
    delete_collection
)
from generate_embeddings import generate_embeddings


# Test database path
TEST_DB_PATH = "data/test_retriever_db"


def cleanup_test_db():
    """Remove test database directory."""
    db_path = Path(TEST_DB_PATH)
    if db_path.exists():
        shutil.rmtree(db_path)


def setup_test_collection():
    """Set up a test collection with sample data."""
    print("Setting up test collection...")

    cleanup_test_db()

    # Initialize database
    client = initialize_db(TEST_DB_PATH)
    collection = get_collection(client, "test_papers")

    # Sample research paper chunks about AI in healthcare
    chunks = [
        "Artificial intelligence is revolutionizing healthcare by enabling early disease detection. Machine learning models can analyze medical images with accuracy comparable to expert radiologists.",
        "Deep learning algorithms have shown remarkable success in diagnosing skin cancer from dermatology images. Studies show 95% accuracy rates in melanoma detection.",
        "Natural language processing helps extract insights from electronic health records. NLP systems can identify patient risks and suggest preventive treatments.",
        "Computer vision in medical imaging can detect subtle patterns invisible to human eyes. This technology improves diagnosis of conditions like pneumonia and tuberculosis.",
        "The weather forecast predicts rain tomorrow. Remember to bring an umbrella when going outside.",
        "Quantum computing may revolutionize drug discovery by simulating molecular interactions. This could accelerate development of new medications."
    ]

    # Generate real embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    # Metadata for each chunk
    metadata = [
        {"page": 1, "source": "ai_healthcare_2024.pdf", "chunk_index": 0},
        {"page": 2, "source": "ai_healthcare_2024.pdf", "chunk_index": 1},
        {"page": 3, "source": "nlp_medicine_2024.pdf", "chunk_index": 0},
        {"page": 1, "source": "medical_imaging_ai.pdf", "chunk_index": 0},
        {"page": 5, "source": "weather_report.pdf", "chunk_index": 0},  # Irrelevant
        {"page": 2, "source": "quantum_computing.pdf", "chunk_index": 0}
    ]

    # Unique IDs
    ids = [f"test_chunk_{i}" for i in range(len(chunks))]

    # Add to collection
    print("Adding documents to collection...")
    add_documents(collection, chunks, embeddings, metadata, ids)

    print(f"✓ Collection set up with {len(chunks)} chunks\n")

    return client, collection


def test_retrieve_relevant_chunks(collection):
    """Test basic retrieval functionality."""
    print("Test 1: Basic retrieval...")

    question = "How is AI used in medical diagnosis?"

    # Retrieve relevant chunks
    results = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        top_k=3,
        distance_threshold=1.0
    )

    # Should return results
    assert len(results) > 0, "Should retrieve at least one chunk"
    assert len(results) <= 3, "Should not exceed top_k limit"

    # Results should have required fields
    for result in results:
        assert 'text' in result, "Result should have text field"
        assert 'page' in result, "Result should have page field"
        assert 'source' in result, "Result should have source field"
        assert 'distance' in result, "Result should have distance field"

    print(f"✓ Retrieved {len(results)} relevant chunks")
    return results


def test_relevance_ranking(collection):
    """Test that most relevant chunks come first."""
    print("\nTest 2: Relevance ranking...")

    question = "How does machine learning help diagnose diseases?"

    results = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        top_k=5
    )

    # First result should be highly relevant (about AI in medical diagnosis)
    first_chunk = results[0]['text'].lower()

    # Check if first result contains relevant keywords
    relevant_keywords = ['machine learning', 'ai', 'artificial intelligence',
                        'disease', 'diagnosis', 'medical', 'healthcare']

    has_relevant_content = any(keyword in first_chunk for keyword in relevant_keywords)
    assert has_relevant_content, "Top result should contain relevant keywords"

    # Results should be sorted by distance (most similar first)
    distances = [r['distance'] for r in results]
    assert distances == sorted(distances), "Results should be sorted by distance"

    print(f"✓ Top result is relevant (distance: {results[0]['distance']:.4f})")


def test_distance_threshold(collection):
    """Test that distance threshold filters irrelevant results."""
    print("\nTest 3: Distance threshold filtering...")

    question = "How does AI help in medical imaging?"

    # Retrieve with strict threshold
    strict_results = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        top_k=10,
        distance_threshold=0.5  # Very strict
    )

    # Retrieve with permissive threshold
    permissive_results = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        top_k=10,
        distance_threshold=2.0  # Very permissive
    )

    # Strict threshold should return fewer results
    assert len(strict_results) <= len(permissive_results), \
        "Strict threshold should filter more results"

    print(f"✓ Strict threshold: {len(strict_results)} results")
    print(f"✓ Permissive threshold: {len(permissive_results)} results")


def test_format_results(collection):
    """Test result formatting."""
    print("\nTest 4: Result formatting...")

    question = "What is AI?"

    from vector_store import search_similar
    from generate_embeddings import generate_single_embedding

    # Get raw ChromaDB results
    query_embedding = generate_single_embedding(question)
    raw_results = search_similar(collection, query_embedding, top_k=2)

    # Format results
    formatted = format_results(raw_results)

    # Check structure
    assert isinstance(formatted, list), "Should return a list"
    assert len(formatted) == 2, "Should format all results"

    # Check fields
    for result in formatted:
        assert 'text' in result
        assert 'page' in result
        assert 'source' in result
        assert 'distance' in result
        assert 'chunk_index' in result
        assert 'id' in result

    print("✓ Results formatted correctly")


def test_build_context(collection):
    """Test context building for LLM."""
    print("\nTest 5: Context building...")

    question = "How is AI used in healthcare?"

    # Retrieve chunks
    chunks = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        top_k=3
    )

    # Build context
    context = build_context(chunks, max_chunks=3)

    # Context should be a non-empty string
    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"

    # Context should include citations
    assert "Source:" in context, "Context should include source citations"
    assert "Page" in context, "Context should include page numbers"

    # Context should include chunk text
    assert chunks[0]['text'][:50] in context, "Context should include chunk text"

    print("✓ Context built successfully")
    print(f"  Context length: {len(context)} characters")


def test_build_prompt(collection):
    """Test complete prompt building."""
    print("\nTest 6: Complete prompt building...")

    question = "What are the applications of AI in medicine?"

    # Retrieve and build context
    chunks = retrieve_relevant_chunks(question, collection, top_k=2)
    context = build_context(chunks)

    # Build complete prompt
    prompt = build_prompt_with_context(question, context)

    # Prompt should include key elements
    assert isinstance(prompt, str), "Prompt should be a string"
    assert question in prompt, "Prompt should include the question"
    assert "Context:" in prompt, "Prompt should have context section"
    assert "Answer:" in prompt, "Prompt should have answer section"

    print("✓ Prompt built successfully")
    print(f"  Prompt length: {len(prompt)} characters")


def test_empty_question_handling():
    """Test handling of invalid inputs."""
    print("\nTest 7: Error handling...")

    # Mock collection (not used in error cases)
    client = initialize_db(TEST_DB_PATH)
    collection = get_collection(client, "test_papers")

    # Empty question should raise error
    try:
        retrieve_relevant_chunks("", collection)
        assert False, "Empty question should raise ValueError"
    except ValueError:
        print("✓ Empty question raises ValueError")

    # Invalid top_k should raise error
    try:
        retrieve_relevant_chunks("test", collection, top_k=0)
        assert False, "Invalid top_k should raise ValueError"
    except ValueError:
        print("✓ Invalid top_k raises ValueError")


def test_reranking(collection):
    """Test keyword-based re-ranking."""
    print("\nTest 8: Keyword-based re-ranking...")

    question = "medical imaging diagnosis"

    # Get initial results
    chunks = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        top_k=5
    )

    if len(chunks) < 2:
        print("⊘ Skipping re-ranking test (not enough results)")
        return

    # Re-rank by keywords
    reranked = rerank_by_keywords(question, chunks)

    # Should return same number of chunks
    assert len(reranked) == len(chunks), "Should return all chunks"

    # Should have combined_score field
    assert 'combined_score' in reranked[0], "Should add combined_score field"

    print("✓ Re-ranking completed successfully")


def test_irrelevant_question(collection):
    """Test behavior with completely irrelevant question."""
    print("\nTest 9: Irrelevant question handling...")

    # Question unrelated to the content
    irrelevant_question = "What is the capital of France?"

    results = retrieve_relevant_chunks(
        question=irrelevant_question,
        collection=collection,
        top_k=3,
        distance_threshold=0.5  # Strict threshold
    )

    # May return few or no results for irrelevant question
    print(f"✓ Returned {len(results)} results for irrelevant question")

    if results:
        print(f"  Best match distance: {results[0]['distance']:.4f}")
        # Distance should be relatively high (not very similar)
        assert results[0]['distance'] > 0.3, "Irrelevant results should have higher distance"


if __name__ == "__main__":

    print("=" * 60)
    print("Running Retriever Tests")
    print("=" * 60 + "\n")

    try:
        # Set up test data
        client, collection = setup_test_collection()

        # Run tests
        test_retrieve_relevant_chunks(collection)
        test_relevance_ranking(collection)
        test_distance_threshold(collection)
        test_format_results(collection)
        test_build_context(collection)
        test_build_prompt(collection)
        test_empty_question_handling()
        test_reranking(collection)
        test_irrelevant_question(collection)

        # Cleanup
        print("\nCleaning up test database...")
        delete_collection(client, "test_papers")
        cleanup_test_db()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        cleanup_test_db()
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_db()
        sys.exit(1)
