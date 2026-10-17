import chromadb
from chromadb.config import Settings
from pathlib import Path


def initialize_db(persist_directory: str = "data/vector_db"):
    """
    Initialize ChromaDB client with persistent storage.

    Args:
        persist_directory: Path where the database will be stored.

    Returns:
        ChromaDB client instance.
    """

    # Convert to absolute path
    db_path = Path(persist_directory).resolve()

    # Create the directory if it doesn't exist
    db_path.mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=str(db_path))

    return client


def get_collection(client, collection_name: str = "research_papers"):
    """
    Get or create a collection in the database.

    Args:
        client: ChromaDB client instance.
        collection_name: Name of the collection.

    Returns:
        ChromaDB collection object.
    """

    # Get existing collection or create a new one
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Research paper chunks with embeddings"}
    )

    return collection


def add_documents(
    collection,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: list[dict],
    ids: list[str]
):
    """
    Add documents with their embeddings to the collection.

    Args:
        collection: ChromaDB collection object.
        chunks: List of text chunks to store.
        embeddings: List of embedding vectors for each chunk.
        metadata: List of metadata dictionaries for each chunk.
        ids: List of unique identifiers for each chunk.
    """

    if not chunks:
        raise ValueError("chunks cannot be empty")

    if len(chunks) != len(embeddings) != len(metadata) != len(ids):
        raise ValueError("All input lists must have the same length")

    # Add documents to the collection
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )


def search_similar(
    collection,
    query_embedding: list[float],
    top_k: int = 5
) -> dict:
    """
    Search for similar chunks using a query embedding.

    Args:
        collection: ChromaDB collection object.
        query_embedding: Embedding vector of the query.
        top_k: Number of similar results to return.

    Returns:
        Dictionary containing documents, distances, metadata, and ids.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


def get_collection_count(collection) -> int:
    """
    Get the number of documents in the collection.

    Args:
        collection: ChromaDB collection object.

    Returns:
        Number of documents in the collection.
    """

    return collection.count()


def delete_collection(client, collection_name: str):
    """
    Delete a collection from the database.

    Args:
        client: ChromaDB client instance.
        collection_name: Name of the collection to delete.
    """

    client.delete_collection(name=collection_name)


def list_collections(client) -> list[str]:
    """
    List all collections in the database.

    Args:
        client: ChromaDB client instance.

    Returns:
        List of collection names.
    """

    collections = client.list_collections()
    return [col.name for col in collections]


if __name__ == "__main__":

    # Test the vector store with sample data
    print("Initializing vector database...")
    client = initialize_db()

    print("Creating collection...")
    collection = get_collection(client, "test_papers")

    # Sample data
    sample_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text."
    ]

    # Generate sample embeddings (normally from generate_embeddings.py)
    # For demo purposes, using dummy embeddings
    sample_embeddings = [[0.1] * 768 for _ in sample_chunks]

    sample_metadata = [
        {"page": 1, "source": "paper1.pdf", "chunk_index": 0},
        {"page": 1, "source": "paper1.pdf", "chunk_index": 1},
        {"page": 2, "source": "paper1.pdf", "chunk_index": 2}
    ]

    sample_ids = ["chunk_0", "chunk_1", "chunk_2"]

    print(f"\nAdding {len(sample_chunks)} documents...")
    add_documents(collection, sample_chunks, sample_embeddings, sample_metadata, sample_ids)

    print(f"Collection now has {get_collection_count(collection)} documents")

    print("\nSearching for similar documents...")
    query_embedding = [0.1] * 768  # Dummy query embedding
    results = search_similar(collection, query_embedding, top_k=2)

    print(f"Found {len(results['documents'][0])} results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n  Result {i+1}:")
        print(f"    Text: {doc[:60]}...")
        print(f"    Metadata: {results['metadatas'][0][i]}")
        print(f"    Distance: {results['distances'][0][i]:.4f}")

    print("\nListing all collections:")
    collections = list_collections(client)
    print(f"  {collections}")

    print("\nCleaning up test collection...")
    delete_collection(client, "test_papers")
    print("Done!")
