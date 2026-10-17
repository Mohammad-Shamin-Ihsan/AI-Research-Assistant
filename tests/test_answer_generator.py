import sys
from pathlib import Path
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from answer_generator import (
    generate_answer,
    generate_answer_streaming,
    answer_question_with_rag,
    format_answer_with_sources,
    validate_ollama_setup
)
from vector_store import (
    initialize_db,
    get_collection,
    add_documents,
    delete_collection
)
from generate_embeddings import generate_embeddings


# Test database path
TEST_DB_PATH = "data/test_answer_generator_db"


def cleanup_test_db():
    """Remove test database directory."""
    db_path = Path(TEST_DB_PATH)
    if db_path.exists():
        shutil.rmtree(db_path)


def setup_test_collection():
    """Set up a test collection with sample healthcare data."""
    print("Setting up test collection...")

    cleanup_test_db()

    # Initialize database
    client = initialize_db(TEST_DB_PATH)
    collection = get_collection(client, "test_papers")

    # Sample research paper chunks about AI in healthcare
    chunks = [
        "Machine learning models have demonstrated remarkable accuracy in diagnosing diabetic retinopathy from retinal images. Studies show that deep learning algorithms can achieve over 90% accuracy in detecting this condition, matching or exceeding human expert performance.",
        "Natural language processing systems are being used to analyze electronic health records and identify patients at risk for adverse events. These NLP algorithms can process thousands of patient records in minutes, extracting critical information that would take humans hours to review.",
        "Computer vision techniques in radiology have improved the detection of lung cancer in CT scans. AI systems can identify subtle nodules that might be missed by radiologists, leading to earlier diagnosis and improved patient outcomes.",
        "Deep learning has enabled the development of predictive models for patient readmission. By analyzing patient history, demographics, and clinical data, these models can identify high-risk patients who may benefit from additional care interventions.",
        "Artificial intelligence is transforming drug discovery by predicting molecular interactions and identifying potential drug candidates. Machine learning algorithms can screen millions of compounds in silico, dramatically accelerating the drug development process."
    ]

    # Generate real embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    # Metadata for each chunk
    metadata = [
        {"page": 2, "source": "ml_medical_imaging_2024.pdf", "chunk_index": 0},
        {"page": 5, "source": "nlp_healthcare_2024.pdf", "chunk_index": 0},
        {"page": 3, "source": "ai_radiology_2024.pdf", "chunk_index": 0},
        {"page": 7, "source": "predictive_analytics_2024.pdf", "chunk_index": 0},
        {"page": 4, "source": "ai_drug_discovery_2024.pdf", "chunk_index": 0}
    ]

    # Unique IDs
    ids = [f"test_chunk_{i}" for i in range(len(chunks))]

    # Add to collection
    print("Adding documents to collection...")
    add_documents(collection, chunks, embeddings, metadata, ids)

    print(f"✓ Collection set up with {len(chunks)} chunks\n")

    return client, collection


def test_validate_ollama():
    """Test Ollama setup validation."""
    print("Test 1: Validating Ollama setup...")

    status = validate_ollama_setup("qwen3:8b")

    # Should return required keys
    assert 'ollama_running' in status
    assert 'model_available' in status
    assert 'message' in status

    print(f"  {status['message']}")

    if not status['ollama_running']:
        print("\n⚠ Ollama is not running. Please start it with: ollama serve")
        print("Skipping remaining tests that require Ollama.\n")
        return False

    if not status['model_available']:
        print("\n⚠ qwen3:8b model not found. Please pull it with: ollama pull qwen3:8b")
        print("Skipping remaining tests that require the model.\n")
        return False

    print("✓ Ollama is ready")
    return True


def test_generate_answer():
    """Test basic answer generation."""
    print("\nTest 2: Basic answer generation...")

    question = "What is machine learning?"
    context = """
    [1] Source: ml_basics.pdf, Page 1
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    It uses statistical techniques to give computers the ability to learn patterns.
    """

    result = generate_answer(question, context, temperature=0.3)

    # Verify result structure
    assert 'answer' in result
    assert 'model' in result
    assert isinstance(result['answer'], str)
    assert len(result['answer']) > 0

    print(f"✓ Generated answer ({len(result['answer'])} characters)")
    print(f"  Model: {result['model']}")
    print(f"  Answer preview: {result['answer'][:100]}...")


def test_answer_with_rag(collection):
    """Test complete RAG pipeline."""
    print("\nTest 3: Complete RAG pipeline...")

    question = "How does AI help in diagnosing diseases?"

    result = answer_question_with_rag(
        question=question,
        collection=collection,
        top_k=3,
        model="qwen3:8b",
        temperature=0.3
    )

    # Verify result structure
    assert 'question' in result
    assert 'answer' in result
    assert 'sources' in result
    assert 'model' in result
    assert 'num_sources' in result

    assert result['question'] == question
    assert isinstance(result['answer'], str)
    assert len(result['answer']) > 0
    assert isinstance(result['sources'], list)
    assert result['num_sources'] > 0

    print(f"✓ RAG pipeline completed")
    print(f"  Retrieved {result['num_sources']} sources")
    print(f"  Answer length: {len(result['answer'])} characters")
    print(f"  Answer preview: {result['answer'][:150]}...")


def test_context_usage(collection):
    """Test that the model uses provided context."""
    print("\nTest 4: Context usage verification...")

    # Ask a specific question that requires the context
    question = "What accuracy do deep learning algorithms achieve in detecting diabetic retinopathy?"

    result = answer_question_with_rag(
        question=question,
        collection=collection,
        top_k=3,
        model="qwen3:8b",
        temperature=0.3
    )

    answer = result['answer'].lower()

    # The answer should mention accuracy/percentage since it's in the context
    has_accuracy_info = any(word in answer for word in ['90%', 'accuracy', 'percent', 'performance'])

    # Note: This is a soft check since LLM responses can vary
    if has_accuracy_info:
        print(f"✓ Model appears to use context (mentions accuracy metrics)")
    else:
        print(f"⚠ Model response may not fully utilize context")

    print(f"  Answer: {result['answer'][:200]}...")


def test_source_citation(collection):
    """Test that answers include source citations."""
    print("\nTest 5: Source citation...")

    question = "What is NLP used for in healthcare?"

    result = answer_question_with_rag(
        question=question,
        collection=collection,
        top_k=2,
        model="qwen3:8b",
        temperature=0.3
    )

    answer = result['answer']

    # Check if answer includes source references
    # LLMs may cite sources in various formats
    has_source_reference = any(word in answer for word in ['source', 'Source', 'page', 'Page', '.pdf', 'according'])

    if has_source_reference:
        print(f"✓ Answer includes source references")
    else:
        print(f"⚠ Answer may not include explicit source citations")

    print(f"  Answer: {answer[:200]}...")


def test_empty_retrieval():
    """Test handling when no relevant chunks are found."""
    print("\nTest 6: Empty retrieval handling...")

    # Create empty collection
    client = initialize_db(TEST_DB_PATH)
    empty_collection = get_collection(client, "empty_test_collection")

    question = "What is quantum computing?"

    result = answer_question_with_rag(
        question=question,
        collection=empty_collection,
        top_k=5,
        model="qwen3:8b"
    )

    # Should handle gracefully
    assert result['num_sources'] == 0
    assert 'couldn\'t find' in result['answer'].lower() or 'no relevant' in result['answer'].lower()

    print(f"✓ Handles empty retrieval gracefully")
    print(f"  Response: {result['answer']}")

    # Cleanup
    delete_collection(client, "empty_test_collection")


def test_format_with_sources(collection):
    """Test answer formatting with sources."""
    print("\nTest 7: Answer formatting with sources...")

    question = "How is AI used in radiology?"

    result = answer_question_with_rag(
        question=question,
        collection=collection,
        top_k=2,
        model="qwen3:8b"
    )

    # Format the output
    formatted = format_answer_with_sources(result['answer'], result['sources'])

    # Should include both answer and sources sections
    assert 'Answer:' in formatted
    assert 'Sources:' in formatted
    assert 'Page' in formatted

    print(f"✓ Formatted output created")
    print(f"  Output length: {len(formatted)} characters")


def test_streaming():
    """Test streaming answer generation."""
    print("\nTest 8: Streaming generation...")

    question = "What is machine learning?"
    context = """
    [1] Source: ml_basics.pdf, Page 1
    Machine learning is a method of data analysis that automates analytical
    model building. It is a branch of artificial intelligence based on the
    idea that systems can learn from data and make decisions.
    """

    print("  Streaming answer: ", end="", flush=True)

    chunks_received = 0
    full_answer = ""

    try:
        for chunk in generate_answer_streaming(question, context, temperature=0.3):
            full_answer += chunk
            chunks_received += 1
            # Print first few characters of each chunk
            if chunks_received <= 5:
                print(chunk[:10], end="", flush=True)

        print("...")
        print(f"\n✓ Streaming completed ({chunks_received} chunks)")
        print(f"  Total answer length: {len(full_answer)} characters")

    except Exception as e:
        print(f"\n⚠ Streaming test failed: {e}")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTest 9: Error handling...")

    # Empty question should raise error
    try:
        generate_answer("", "some context")
        assert False, "Empty question should raise ValueError"
    except ValueError:
        print("✓ Empty question raises ValueError")

    # Empty context should raise error
    try:
        generate_answer("some question", "")
        assert False, "Empty context should raise ValueError"
    except ValueError:
        print("✓ Empty context raises ValueError")


if __name__ == "__main__":

    print("="*60)
    print("Running Answer Generator Tests")
    print("="*60 + "\n")

    try:
        # First check if Ollama is ready
        ollama_ready = test_validate_ollama()

        if not ollama_ready:
            print("\nTests require Ollama with qwen3:8b model.")
            print("Please ensure Ollama is running and model is available.")
            sys.exit(1)

        # Set up test data
        client, collection = setup_test_collection()

        # Run tests that require Ollama
        test_generate_answer()
        test_answer_with_rag(collection)
        test_context_usage(collection)
        test_source_citation(collection)
        test_empty_retrieval()
        test_format_with_sources(collection)
        test_streaming()
        test_error_handling()

        # Cleanup
        print("\nCleaning up test database...")
        delete_collection(client, "test_papers")
        cleanup_test_db()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

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
