import sys
from pathlib import Path
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_store import (
    initialize_db,
    get_collection,
    add_documents,
    search_similar,
    get_collection_count,
    delete_collection,
    list_collections
)
from generate_embeddings import generate_embeddings


# Test database path
TEST_DB_PATH = "data/test_vector_db"


def cleanup_test_db():
    """Remove test database directory."""
    db_path = Path(TEST_DB_PATH)
    if db_path.exists():
        shutil.rmtree(db_path)


def test_initialize_db():
    """Test database initialization."""
    print("Test 1: Database initialization...")

    cleanup_test_db()

    client = initialize_db(TEST_DB_PATH)
    assert client is not None, "Client should be initialized"

    # Check if directory was created
    db_path = Path(TEST_DB_PATH)
    assert db_path.exists(), "Database directory should exist"

    print("✓ Database initialized successfully")
    return client


def test_create_collection(client):
    """Test collection creation."""
    print("\nTest 2: Collection creation...")

    collection = get_collection(client, "test_collection")
    assert collection is not None, "Collection should be created"
    assert collection.name == "test_collection", "Collection name should match"

    # Verify collection appears in list
    collections = list_collections(client)
    assert "test_collection" in collections, "Collection should be in list"

    print("✓ Collection created successfully")
    return collection


def test_add_documents(collection):
    """Test adding documents to collection."""
    print("\nTest 3: Adding documents...")

    # Sample chunks
    chunks = [
        "Artificial intelligence is transforming healthcare.",
        "Machine learning models can detect diseases early.",
        "Deep learning requires large amounts of data."
    ]

    # Generate real embeddings
    embeddings = generate_embeddings(chunks)

    # Metadata for each chunk
    metadata = [
        {"page": 1, "source": "test_paper.pdf", "chunk_index": 0},
        {"page": 1, "source": "test_paper.pdf", "chunk_index": 1},
        {"page": 2, "source": "test_paper.pdf", "chunk_index": 2}
    ]

    # Unique IDs
    ids = [f"test_chunk_{i}" for i in range(len(chunks))]

    # Add documents
    add_documents(collection, chunks, embeddings, metadata, ids)

    # Verify count
    count = get_collection_count(collection)
    assert count == len(chunks), f"Expected {len(chunks)} documents, got {count}"

    print(f"✓ Added {count} documents successfully")
    return chunks, embeddings


def test_search_similar(collection, chunks, embeddings):
    """Test searching for similar documents."""
    print("\nTest 4: Searching similar documents...")

    # Use the first chunk's embedding as query
    query_embedding = embeddings[0]

    # Search for top 2 similar documents
    results = search_similar(collection, query_embedding, top_k=2)

    # Verify results structure
    assert 'documents' in results, "Results should contain documents"
    assert 'metadatas' in results, "Results should contain metadatas"
    assert 'distances' in results, "Results should contain distances"
    assert 'ids' in results, "Results should contain ids"

    # Verify number of results
    num_results = len(results['documents'][0])
    assert num_results == 2, f"Expected 2 results, got {num_results}"

    # First result should be the query itself (distance ~0)
    first_distance = results['distances'][0][0]
    assert first_distance < 0.01, f"First result should be very similar (distance: {first_distance})"

    print(f"✓ Search returned {num_results} results")
    print(f"  First result distance: {first_distance:.6f}")
    print(f"  Second result distance: {results['distances'][0][1]:.6f}")


def test_semantic_search(collection):
    """Test that semantic search finds relevant content."""
    print("\nTest 5: Semantic search relevance...")

    # Query about AI and healthcare
    query_text = "How is AI used in medical diagnosis?"
    query_embedding = generate_embeddings([query_text])[0]

    # Search
    results = search_similar(collection, query_embedding, top_k=1)

    # The most relevant chunk should be about AI in healthcare
    top_result = results['documents'][0][0]
    print(f"  Query: {query_text}")
    print(f"  Top result: {top_result}")

    # Check if result is relevant (contains AI/healthcare keywords)
    relevant_keywords = ['artificial', 'intelligence', 'healthcare', 'medical', 'disease']
    is_relevant = any(keyword in top_result.lower() for keyword in relevant_keywords)

    assert is_relevant, "Top result should be semantically relevant"
    print("✓ Semantic search found relevant content")


def test_metadata_preservation(collection):
    """Test that metadata is correctly stored and retrieved."""
    print("\nTest 6: Metadata preservation...")

    # Search and get results
    query_embedding = generate_embeddings(["test query"])[0]
    results = search_similar(collection, query_embedding, top_k=1)

    # Check metadata
    metadata = results['metadatas'][0][0]

    assert 'page' in metadata, "Metadata should contain page"
    assert 'source' in metadata, "Metadata should contain source"
    assert 'chunk_index' in metadata, "Metadata should contain chunk_index"

    print(f"✓ Metadata preserved correctly: {metadata}")


def test_empty_input_handling(collection):
    """Test handling of invalid inputs."""
    print("\nTest 7: Error handling...")

    # Empty chunks should raise error
    try:
        add_documents(collection, [], [], [], [])
        assert False, "Empty chunks should raise ValueError"
    except ValueError:
        print("✓ Empty input raises ValueError")

    # Mismatched lengths should raise error
    try:
        add_documents(
            collection,
            chunks=["test"],
            embeddings=[[0.1] * 768],
            metadata=[{"page": 1}],
            ids=["id1", "id2"]  # Wrong length
        )
        assert False, "Mismatched lengths should raise ValueError"
    except ValueError:
        print("✓ Mismatched lengths raise ValueError")

    # Invalid top_k should raise error
    try:
        search_similar(collection, [0.1] * 768, top_k=0)
        assert False, "Invalid top_k should raise ValueError"
    except ValueError:
        print("✓ Invalid top_k raises ValueError")


def test_collection_deletion(client):
    """Test deleting a collection."""
    print("\nTest 8: Collection deletion...")

    # Get current collections
    collections_before = list_collections(client)
    assert "test_collection" in collections_before, "Test collection should exist"

    # Delete collection
    delete_collection(client, "test_collection")

    # Verify deletion
    collections_after = list_collections(client)
    assert "test_collection" not in collections_after, "Test collection should be deleted"

    print("✓ Collection deleted successfully")


if __name__ == "__main__":

    print("=" * 60)
    print("Running Vector Store Tests")
    print("=" * 60 + "\n")

    try:
        # Run tests in sequence
        client = test_initialize_db()
        collection = test_create_collection(client)
        chunks, embeddings = test_add_documents(collection)
        test_search_similar(collection, chunks, embeddings)
        test_semantic_search(collection)
        test_metadata_preservation(collection)
        test_empty_input_handling(collection)
        test_collection_deletion(client)

        # Cleanup
        print("\nCleaning up test database...")
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
        cleanup_test_db()
        sys.exit(1)
