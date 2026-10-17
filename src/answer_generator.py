import ollama
from retriever import (
    retrieve_relevant_chunks,
    build_context,
    build_prompt_with_context
)


def generate_answer(
    question: str,
    context: str,
    model: str = "qwen3:8b",
    temperature: float = 0.3,
    max_tokens: int = 1000
) -> dict:
    """
    Generate an answer using Qwen3:8b based on the provided context.

    This function sends the context and question to the Qwen3:8b model
    through Ollama and returns the generated answer.

    Args:
        question: The user's question.
        context: The relevant context from retrieved chunks.
        model: Name of the Ollama model to use.
        temperature: Controls randomness (0.0-1.0).
                    Lower = more focused/deterministic (better for factual Q&A).
                    Higher = more creative/varied.
        max_tokens: Maximum number of tokens to generate in the response.

    Returns:
        A dictionary containing:
        - answer: The generated answer text
        - model: The model used
        - prompt_tokens: Approximate tokens in the prompt
        - response_tokens: Approximate tokens in the response
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty")

    # Build the complete prompt with instructions
    # This structures the input to encourage proper citation and accuracy
    prompt = build_prompt_with_context(question, context)

    try:
        # Send request to Ollama using the chat API
        # Using chat format allows us to set a system prompt for better instruction following
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a helpful research assistant. Answer questions based ONLY on '
                        'the provided context from research papers. Always cite your sources using '
                        'the format [Source: filename, Page X]. If the context does not contain '
                        'enough information to answer the question, clearly state that. '
                        'Be accurate, concise, and cite specific sources for each claim.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': temperature,
                'num_predict': max_tokens  # Maximum tokens to generate
            }
        )

        # Extract the answer from the response
        answer = response['message']['content']

        # Extract token usage information if available
        # Note: Ollama may not always provide exact token counts
        prompt_tokens = response.get('prompt_eval_count', 0)
        response_tokens = response.get('eval_count', 0)

        return {
            'answer': answer,
            'model': model,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens
        }

    except Exception as e:
        # Provide helpful error messages for common issues
        error_msg = str(e).lower()

        if 'connection' in error_msg or 'connect' in error_msg:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure Ollama is running.\n"
                "Start it with: ollama serve"
            )
        elif 'model' in error_msg and 'not found' in error_msg:
            raise RuntimeError(
                f"Model '{model}' not found. Pull it with: ollama pull {model}"
            )
        else:
            raise RuntimeError(f"Error generating answer: {e}")


def generate_answer_streaming(
    question: str,
    context: str,
    model: str = "qwen3:8b",
    temperature: float = 0.3
):
    """
    Generate an answer with streaming output (yields chunks as generated).

    This provides a better user experience by showing the answer as it's
    being generated, rather than waiting for the complete response.

    Args:
        question: The user's question.
        context: The relevant context from retrieved chunks.
        model: Name of the Ollama model to use.
        temperature: Controls randomness (0.0-1.0).

    Yields:
        Chunks of the generated answer as they become available.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty")

    prompt = build_prompt_with_context(question, context)

    try:
        # Stream the response from Ollama
        stream = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a helpful research assistant. Answer questions based ONLY on '
                        'the provided context from research papers. Always cite your sources using '
                        'the format [Source: filename, Page X]. If the context does not contain '
                        'enough information to answer the question, clearly state that. '
                        'Be accurate, concise, and cite specific sources for each claim.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': temperature
            },
            stream=True  # Enable streaming
        )

        # Yield each chunk as it arrives
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']

    except Exception as e:
        error_msg = str(e).lower()

        if 'connection' in error_msg or 'connect' in error_msg:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure Ollama is running.\n"
                "Start it with: ollama serve"
            )
        elif 'model' in error_msg and 'not found' in error_msg:
            raise RuntimeError(
                f"Model '{model}' not found. Pull it with: ollama pull {model}"
            )
        else:
            raise RuntimeError(f"Error generating answer: {e}")


def answer_question_with_rag(
    question: str,
    collection,
    top_k: int = 5,
    model: str = "qwen3:8b",
    temperature: float = 0.3,
    distance_threshold: float = 1.0
) -> dict:
    """
    Complete Retrieval-Augmented Generation (RAG) pipeline.

    This is the main function that ties everything together:
    1. Retrieves relevant chunks from the vector database
    2. Builds context from retrieved chunks
    3. Generates an answer using Qwen3:8b
    4. Returns the answer with source citations

    Args:
        question: The user's question.
        collection: ChromaDB collection to search.
        top_k: Number of chunks to retrieve.
        model: Name of the Ollama model to use.
        temperature: Controls randomness in generation.
        distance_threshold: Maximum distance for relevant chunks.

    Returns:
        A dictionary containing:
        - question: The original question
        - answer: The generated answer
        - sources: List of source chunks used
        - model: The model used
        - num_sources: Number of sources retrieved
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    # Step 1: Retrieve relevant chunks from the vector database
    retrieved_chunks = retrieve_relevant_chunks(
        question=question,
        collection=collection,
        top_k=top_k,
        distance_threshold=distance_threshold
    )

    # Handle case where no relevant chunks are found
    if not retrieved_chunks:
        return {
            'question': question,
            'answer': (
                "I couldn't find any relevant information in the knowledge base "
                "to answer your question. Please try rephrasing your question or "
                "ask about a different topic."
            ),
            'sources': [],
            'model': model,
            'num_sources': 0
        }

    # Step 2: Build context from retrieved chunks
    context = build_context(retrieved_chunks, max_chunks=top_k)

    # Step 3: Generate answer using Qwen3:8b
    result = generate_answer(
        question=question,
        context=context,
        model=model,
        temperature=temperature
    )

    # Step 4: Format and return the complete response
    return {
        'question': question,
        'answer': result['answer'],
        'sources': retrieved_chunks,
        'model': result['model'],
        'num_sources': len(retrieved_chunks),
        'prompt_tokens': result.get('prompt_tokens', 0),
        'response_tokens': result.get('response_tokens', 0)
    }


def format_answer_with_sources(answer: str, retrieved_chunks: list[dict]) -> str:
    """
    Format the final response with answer and source references.

    Creates a nicely formatted output that includes:
    - The generated answer
    - A list of sources used with page numbers
    - Relevance scores for each source

    Args:
        answer: The generated answer text.
        retrieved_chunks: List of chunks that were used as context.

    Returns:
        A formatted string with answer and sources.
    """

    # Start with the answer
    formatted = f"Answer:\n{answer}\n"

    # Add sources section if any chunks were used
    if retrieved_chunks:
        formatted += "\n" + "="*60 + "\n"
        formatted += "Sources:\n"
        formatted += "="*60 + "\n"

        for i, chunk in enumerate(retrieved_chunks, start=1):
            formatted += f"\n[{i}] {chunk['source']} (Page {chunk['page']})\n"
            formatted += f"    Relevance: {1 - chunk['distance']:.2%}\n"
            formatted += f"    Preview: {chunk['text'][:100]}...\n"

    return formatted


def validate_ollama_setup(model: str = "qwen3:8b") -> dict:
    """
    Validate that Ollama is running and the model is available.

    This is useful for debugging and setup verification.

    Args:
        model: Name of the model to check.

    Returns:
        A dictionary with status information:
        - ollama_running: bool
        - model_available: bool
        - message: str
    """

    try:
        # Try to list available models
        models = ollama.list()

        # Check if our model is in the list
        model_names = [m['name'] for m in models.get('models', [])]
        model_available = any(model in name for name in model_names)

        if model_available:
            return {
                'ollama_running': True,
                'model_available': True,
                'message': f"✓ Ollama is running and {model} is available"
            }
        else:
            return {
                'ollama_running': True,
                'model_available': False,
                'message': f"✓ Ollama is running but {model} not found. Pull it with: ollama pull {model}"
            }

    except Exception as e:
        return {
            'ollama_running': False,
            'model_available': False,
            'message': f"✗ Cannot connect to Ollama: {e}\nMake sure Ollama is running: ollama serve"
        }


if __name__ == "__main__":

    print("="*60)
    print("Answer Generator Example")
    print("="*60 + "\n")

    # First, validate Ollama setup
    print("Validating Ollama setup...")
    status = validate_ollama_setup("qwen3:8b")
    print(status['message'])
    print()

    if not status['ollama_running']:
        print("Please start Ollama and try again.")
        exit(1)

    if not status['model_available']:
        print("Please pull the model first: ollama pull qwen3:8b")
        exit(1)

    # Example: Using the complete RAG pipeline
    from vector_store import initialize_db, get_collection, get_collection_count

    print("Initializing database...")
    client = initialize_db()
    collection = get_collection(client, "research_papers")

    count = get_collection_count(collection)

    if count == 0:
        print("\nWarning: Collection is empty.")
        print("\nTo populate the database, you need to:")
        print("1. Extract text from PDFs (extract_pdf.py)")
        print("2. Chunk the text (chunk_text.py)")
        print("3. Generate embeddings (generate_embeddings.py)")
        print("4. Store in vector database (vector_store.py)")
        print("\nFor now, here's an example of direct answer generation:")

        # Example with mock context
        sample_question = "How does machine learning improve healthcare?"
        sample_context = """
        [1] Source: ai_healthcare.pdf, Page 3 (relevance score: 0.850)
        Machine learning algorithms have revolutionized medical diagnosis by
        analyzing patient data and identifying patterns that humans might miss.
        These systems can predict disease onset and recommend treatments.
        """

        print(f"\nQuestion: {sample_question}")
        print(f"\nContext: {sample_context}")
        print("\nGenerating answer...")

        result = generate_answer(sample_question, sample_context)

        print("\n" + "="*60)
        print("Generated Answer:")
        print("="*60)
        print(result['answer'])
        print(f"\nModel: {result['model']}")
        print(f"Tokens: {result['prompt_tokens']} prompt, {result['response_tokens']} response")

    else:
        print(f"Found {count} chunks in database\n")

        # Example question
        question = "What are the applications of artificial intelligence in medicine?"
        print(f"Question: {question}\n")

        print("Running complete RAG pipeline...")
        print("1. Retrieving relevant chunks...")
        print("2. Building context...")
        print("3. Generating answer with Qwen3:8b...\n")

        # Run the complete RAG pipeline
        result = answer_question_with_rag(
            question=question,
            collection=collection,
            top_k=3,
            model="qwen3:8b"
        )

        # Display the formatted result
        print("="*60)
        print("Result")
        print("="*60 + "\n")

        formatted_output = format_answer_with_sources(
            result['answer'],
            result['sources']
        )

        print(formatted_output)

        print(f"\nModel: {result['model']}")
        print(f"Sources used: {result['num_sources']}")
