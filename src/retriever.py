from generate_embeddings import generate_single_embedding
from vector_store import search_similar


def retrieve_relevant_chunks(
    question: str,
    collection,
    top_k: int = 5,
    distance_threshold: float = 1.0
) -> list[dict]:
    """
    Find the most relevant chunks for a user's question.

    This function performs semantic search by:
    1. Converting the question into an embedding vector
    2. Searching the vector database for similar chunks
    3. Filtering results based on distance threshold
    4. Formatting results for easy consumption

    Args:
        question: The user's question as a string.
        collection: ChromaDB collection to search in.
        top_k: Maximum number of results to return.
        distance_threshold: Maximum distance to consider a chunk relevant.
                          Lower = more similar. Typical range: 0.0-2.0
                          Default 1.0 is fairly permissive.

    Returns:
        A list of dictionaries, each containing:
        - text: The chunk text
        - page: Page number from source document
        - source: Source filename
        - distance: Similarity score (lower is more similar)
        - chunk_index: Position of chunk in original document
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    # Step 1: Convert the question into an embedding vector
    # This allows us to compare the question semantically with stored chunks
    question_embedding = generate_single_embedding(question)

    # Step 2: Search the vector database for similar chunks
    # ChromaDB will find chunks with embeddings closest to our question
    results = search_similar(collection, question_embedding, top_k=top_k)

    # Step 3: Format the results into a cleaner structure
    formatted_chunks = format_results(results)

    # Step 4: Filter out chunks that are too dissimilar
    # A high distance score means the chunk is not semantically related
    filtered_chunks = [
        chunk for chunk in formatted_chunks
        if chunk['distance'] <= distance_threshold
    ]

    return filtered_chunks


def format_results(results: dict) -> list[dict]:
    """
    Convert ChromaDB query results into a clean, readable format.

    ChromaDB returns results in a nested structure with parallel arrays.
    This function transforms them into a list of dictionaries for easier use.

    Args:
        results: Raw results from ChromaDB query containing:
                - documents: List of text chunks
                - metadatas: List of metadata dictionaries
                - distances: List of similarity scores
                - ids: List of unique identifiers

    Returns:
        A list of dictionaries, one per retrieved chunk, with all
        relevant information combined.
    """

    # ChromaDB wraps results in an extra list dimension
    # Extract the actual data from the first (and only) query result
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    ids = results['ids'][0]

    # Combine parallel arrays into a list of dictionaries
    # This makes it much easier to work with individual results
    formatted = []
    for doc, metadata, distance, chunk_id in zip(documents, metadatas, distances, ids):
        formatted.append({
            'text': doc,
            'page': metadata.get('page', 'Unknown'),
            'source': metadata.get('source', 'Unknown'),
            'distance': distance,
            'chunk_index': metadata.get('chunk_index', 0),
            'id': chunk_id
        })

    return formatted


def build_context(retrieved_chunks: list[dict], max_chunks: int = 5) -> str:
    """
    Combine retrieved chunks into a single context string for the LLM.

    This function creates a formatted context that includes:
    - Source citations for each chunk
    - Clear separation between chunks
    - Limited number of chunks to fit in LLM context window

    Args:
        retrieved_chunks: List of chunk dictionaries from retrieve_relevant_chunks.
        max_chunks: Maximum number of chunks to include in context.

    Returns:
        A formatted string ready to be passed to the LLM as context.
    """

    if not retrieved_chunks:
        return "No relevant information found in the knowledge base."

    # Limit the number of chunks to prevent overwhelming the LLM context
    chunks_to_use = retrieved_chunks[:max_chunks]

    context_parts = []

    # Build context with clear citations
    for i, chunk in enumerate(chunks_to_use, start=1):
        # Format: [1] Content from source, page X (distance: 0.XX)
        citation = f"[{i}] Source: {chunk['source']}, Page {chunk['page']}"

        # Include distance score for debugging/transparency
        # Lower distance = more relevant
        citation += f" (relevance score: {chunk['distance']:.3f})"

        # Combine citation with actual text content
        chunk_text = f"{citation}\n{chunk['text']}"

        context_parts.append(chunk_text)

    # Join all chunks with clear separators
    context = "\n\n---\n\n".join(context_parts)

    return context


def build_prompt_with_context(question: str, context: str) -> str:
    """
    Create a complete prompt for the LLM combining context and question.

    This structures the prompt to encourage the LLM to:
    - Use only the provided context
    - Cite sources when answering
    - Admit when information is insufficient

    Args:
        question: The user's question.
        context: The relevant context from retrieved chunks.

    Returns:
        A complete prompt string ready for the LLM.
    """

    prompt = f"""Based on the following context from research papers, answer the question.
If the context doesn't contain enough information to answer the question, say so.
Always cite the source and page number when providing information.

Context:
{context}

Question: {question}

Answer:"""

    return prompt


def rerank_by_keywords(question: str, chunks: list[dict]) -> list[dict]:
    """
    Re-rank retrieved chunks based on keyword overlap with the question.

    This is an optional enhancement to pure vector similarity.
    Chunks that contain more question keywords may be more relevant,
    even if vector similarity is slightly lower.

    Args:
        question: The user's question.
        chunks: List of retrieved chunks to re-rank.

    Returns:
        Re-ranked list of chunks (sorted by combined score).
    """

    # Extract meaningful words from question (lowercase, remove common words)
    stop_words = {'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'does', 'do'}

    question_words = set(
        word.lower().strip('.,!?;:')
        for word in question.split()
        if word.lower() not in stop_words and len(word) > 2
    )

    # Score each chunk based on keyword overlap
    for chunk in chunks:
        text_lower = chunk['text'].lower()

        # Count how many question keywords appear in this chunk
        keyword_matches = sum(1 for word in question_words if word in text_lower)

        # Combine vector distance with keyword score
        # Lower distance is better, higher keyword matches is better
        # Weight both factors (you can adjust these weights)
        combined_score = chunk['distance'] - (keyword_matches * 0.1)

        chunk['combined_score'] = combined_score

    # Sort by combined score (lower is better)
    reranked = sorted(chunks, key=lambda x: x['combined_score'])

    return reranked


if __name__ == "__main__":

    # Example usage demonstrating the retrieval pipeline
    print("=" * 60)
    print("Retrieval System Example")
    print("=" * 60 + "\n")

    # This example assumes you have a populated vector database
    # In practice, you would:
    # 1. Load your ChromaDB client and collection
    # 2. Call retrieve_relevant_chunks with a real question
    # 3. Use the results to answer questions

    from vector_store import initialize_db, get_collection

    # Initialize database (assumes it's already populated)
    print("Initializing database...")
    client = initialize_db()
    collection = get_collection(client, "research_papers")

    # Check if collection has data
    from vector_store import get_collection_count
    count = get_collection_count(collection)

    if count == 0:
        print("Warning: Collection is empty. Please add documents first.")
        print("\nTo populate the database:")
        print("1. Extract text from PDFs (extract_pdf.py)")
        print("2. Chunk the text (chunk_text.py)")
        print("3. Generate embeddings (generate_embeddings.py)")
        print("4. Store in vector database (vector_store.py)")
    else:
        print(f"Found {count} chunks in database\n")

        # Example question
        question = "How does machine learning improve medical diagnosis?"
        print(f"Question: {question}\n")

        # Retrieve relevant chunks
        print("Retrieving relevant chunks...")
        relevant_chunks = retrieve_relevant_chunks(
            question=question,
            collection=collection,
            top_k=3,
            distance_threshold=0.8
        )

        print(f"Found {len(relevant_chunks)} relevant chunks\n")

        # Display results
        for i, chunk in enumerate(relevant_chunks, start=1):
            print(f"--- Chunk {i} ---")
            print(f"Source: {chunk['source']}, Page {chunk['page']}")
            print(f"Relevance Score: {chunk['distance']:.4f}")
            print(f"Text: {chunk['text'][:150]}...")
            print()

        # Build context for LLM
        print("\n" + "=" * 60)
        print("Building LLM Context")
        print("=" * 60 + "\n")

        context = build_context(relevant_chunks, max_chunks=3)
        print(context)

        # Build complete prompt
        print("\n" + "=" * 60)
        print("Complete LLM Prompt")
        print("=" * 60 + "\n")

        prompt = build_prompt_with_context(question, context)
        print(prompt)
