# AI Research Paper Assistant

A fully local, privacy-focused AI assistant for analyzing research papers using Retrieval-Augmented Generation (RAG). Ask questions about your research papers and get accurate answers with source citations.

## ✨ Features

- 📄 **PDF Text Extraction** - Extract text from research papers while preserving page numbers
- 🔤 **Smart Text Chunking** - Split documents into semantic chunks with overlap for better context
- 🧠 **Semantic Search** - Find relevant passages using vector embeddings
- 💬 **Question Answering** - Get accurate answers from your papers using Qwen3:8b LLM
- 🔒 **100% Local** - No API calls, complete privacy, zero costs
- 📚 **Source Citations** - Every answer includes source files and page numbers
- ⚡ **Fast Retrieval** - ChromaDB vector database for efficient similarity search

## 🏗️ Architecture

```
PDF → Text Extraction → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer
       (PyMuPDF)                    (Ollama)    (ChromaDB)            (Qwen3:8b)
```

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
- **Ollama** - [Install Ollama](https://ollama.ai/download)
- **Git** - [Install Git](https://git-scm.com/downloads)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mohammad-Shamin-Ihsan/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama Models

```bash
# Install embedding model (for converting text to vectors)
ollama pull nomic-embed-text

# Install LLM model (for generating answers)
ollama pull qwen3:8b
```

This will download approximately:
- `nomic-embed-text`: ~274 MB
- `qwen3:8b`: ~4.7 GB

### Step 5: Verify Installation

```bash
# Start Ollama service (if not already running)
ollama serve

# In a new terminal, test the setup
python -c "from src.answer_generator import validate_ollama_setup; print(validate_ollama_setup()['message'])"
```

You should see: `✓ Ollama is running and qwen3:8b is available`

## 📖 Usage Guide

### Quick Start

Here's a complete example of using the system:

```python
from src.extract_pdf import extract_text_from_pdf
from src.chunk_text import chunk_text
from src.generate_embeddings import generate_embeddings
from src.vector_store import initialize_db, get_collection, add_documents
from src.answer_generator import answer_question_with_rag

# 1. Extract text from your PDF
pdf_path = "data/papers/your_paper.pdf"
pages = extract_text_from_pdf(pdf_path)

# 2. Combine all pages and chunk the text
full_text = " ".join([page["text"] for page in pages])
chunks = chunk_text(full_text, chunk_size=1500, overlap=200)

# 3. Generate embeddings for chunks
embeddings = generate_embeddings(chunks)

# 4. Store in vector database
client = initialize_db()
collection = get_collection(client, "research_papers")

# Prepare metadata
metadata = [
    {"page": i // 3 + 1, "source": "your_paper.pdf", "chunk_index": i}
    for i in range(len(chunks))
]
ids = [f"chunk_{i}" for i in range(len(chunks))]

add_documents(collection, chunks, embeddings, metadata, ids)

# 5. Ask questions!
result = answer_question_with_rag(
    question="What is the main contribution of this paper?",
    collection=collection,
    top_k=5
)

print(result['answer'])
print(f"\nSources used: {result['num_sources']}")
```

### Step-by-Step Workflow

#### 1. Add Your PDF Papers

Place your research papers in the `data/papers/` directory:

```bash
mkdir -p data/papers
cp /path/to/your/paper.pdf data/papers/
```

#### 2. Extract Text from PDF

```python
from src.extract_pdf import extract_text_from_pdf

# Extract text page by page
pages = extract_text_from_pdf("data/papers/your_paper.pdf")

print(f"Extracted {len(pages)} pages")
print(f"First page preview: {pages[0]['text'][:200]}")
```

#### 3. Chunk the Text

```python
from src.chunk_text import chunk_text

# Combine all pages
full_text = " ".join([page["text"] for page in pages])

# Split into chunks
chunks = chunk_text(
    text=full_text,
    chunk_size=1500,  # Characters per chunk
    overlap=200       # Overlap between chunks
)

print(f"Created {len(chunks)} chunks")
```

#### 4. Generate Embeddings

```python
from src.generate_embeddings import generate_embeddings

# Convert chunks to vector embeddings
embeddings = generate_embeddings(chunks)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")  # Should be 768
```

#### 5. Store in Vector Database

```python
from src.vector_store import initialize_db, get_collection, add_documents

# Initialize database
client = initialize_db("data/vector_db")
collection = get_collection(client, "research_papers")

# Prepare metadata for each chunk
metadata = [
    {
        "page": (i // 3) + 1,  # Approximate page number
        "source": "your_paper.pdf",
        "chunk_index": i
    }
    for i in range(len(chunks))
]

# Generate unique IDs
ids = [f"paper1_chunk_{i}" for i in range(len(chunks))]

# Store everything
add_documents(collection, chunks, embeddings, metadata, ids)

print(f"Stored {len(chunks)} chunks in vector database")
```

#### 6. Ask Questions

```python
from src.answer_generator import answer_question_with_rag, format_answer_with_sources

# Ask a question
question = "What methodology does this paper use?"

result = answer_question_with_rag(
    question=question,
    collection=collection,
    top_k=5,  # Number of relevant chunks to retrieve
    model="qwen3:8b",
    temperature=0.3  # Lower = more focused answers
)

# Display formatted answer with sources
output = format_answer_with_sources(result['answer'], result['sources'])
print(output)
```

### Using Individual Modules

#### Extract PDF Only

```python
from src.extract_pdf import extract_text_from_pdf

pages = extract_text_from_pdf("data/papers/paper.pdf")

# Access specific page
print(f"Page 1: {pages[0]['text']}")
```

#### Chunk Text Only

```python
from src.chunk_text import chunk_text

text = "Your long text here..."
chunks = chunk_text(text, chunk_size=1000, overlap=100)
```

#### Generate Single Embedding

```python
from src.generate_embeddings import generate_single_embedding

embedding = generate_single_embedding("Sample text to embed")
print(f"Embedding dimension: {len(embedding)}")
```

#### Search Vector Database

```python
from src.retriever import retrieve_relevant_chunks

chunks = retrieve_relevant_chunks(
    question="What is machine learning?",
    collection=collection,
    top_k=3,
    distance_threshold=0.8
)

for chunk in chunks:
    print(f"Source: {chunk['source']}, Page: {chunk['page']}")
    print(f"Text: {chunk['text'][:100]}...")
    print(f"Relevance: {1 - chunk['distance']:.2%}\n")
```

## 🧪 Running Tests

The project includes comprehensive tests for all modules:

```bash
# Test PDF extraction
python tests/test_vector_store.py

# Test embeddings generation
python tests/test_embeddings.py

# Test retrieval system
python tests/test_retriever.py

# Test answer generation (requires Ollama)
python tests/test_answer_generator.py
```

## 📁 Project Structure

```
AI-Research-Assistant/
├── src/
│   ├── extract_pdf.py          # PDF text extraction
│   ├── chunk_text.py           # Text chunking with overlap
│   ├── generate_embeddings.py  # Embedding generation via Ollama
│   ├── vector_store.py         # ChromaDB vector database
│   ├── retriever.py            # Semantic search and retrieval
│   └── answer_generator.py     # Answer generation with Qwen3:8b
├── tests/
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   └── test_answer_generator.py
├── data/
│   ├── papers/                 # Place your PDF files here
│   └── vector_db/              # Vector database storage
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Adjusting Chunk Size

Edit chunk parameters based on your needs:

```python
chunks = chunk_text(
    text=full_text,
    chunk_size=1500,  # Larger = more context per chunk
    overlap=200       # Larger = more context preservation
)
```

**Recommendations:**
- **Technical papers**: 1500-2000 characters
- **General content**: 1000-1500 characters
- **Overlap**: 15-20% of chunk_size

### Tuning Retrieval

Adjust retrieval parameters:

```python
result = answer_question_with_rag(
    question=question,
    collection=collection,
    top_k=5,              # More = more context (slower)
    temperature=0.3,      # Lower = more focused (0.1-0.9)
    distance_threshold=1.0  # Lower = stricter relevance
)
```

### Using Different Models

You can use other Ollama models:

```python
# List available models
import ollama
models = ollama.list()
print(models)

# Use a different LLM
result = answer_question_with_rag(
    question=question,
    collection=collection,
    model="llama3.1:8b"  # Or any other model
)
```

## 🔧 Troubleshooting

### Ollama Connection Error

```
Error: Cannot connect to Ollama
```

**Solution:**
```bash
# Start Ollama service
ollama serve

# Or on macOS (if installed via app)
# Ollama should auto-start
```

### Model Not Found

```
Error: Model 'qwen3:8b' not found
```

**Solution:**
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### Out of Memory Error

If you encounter memory issues:

1. **Reduce chunk size**: Use smaller chunks
2. **Reduce top_k**: Retrieve fewer chunks
3. **Use smaller model**: Try `qwen2.5:3b` instead of `qwen3:8b`

```bash
ollama pull qwen2.5:3b
```

### PDF Extraction Issues

If PDF extraction fails:

- Ensure the PDF is not encrypted
- Some scanned PDFs may need OCR (not included)
- Try re-saving the PDF using a PDF viewer

### Slow Performance

**For faster embeddings:**
- GPU acceleration is automatic if available
- Process PDFs in batches
- Use multiprocessing for large collections

**For faster answers:**
- Reduce `top_k` (retrieve fewer chunks)
- Use a smaller model like `qwen2.5:3b`

## 💡 Tips & Best Practices

### 1. Organizing Multiple Papers

```python
# Use descriptive collection names
collection = get_collection(client, "machine_learning_papers")

# Include paper metadata
metadata = {
    "page": page_num,
    "source": "paper_name.pdf",
    "authors": "Smith et al.",
    "year": 2024,
    "chunk_index": i
}
```

### 2. Better Question Phrasing

❌ **Vague:** "Tell me about this paper"  
✅ **Specific:** "What methodology does this paper use for image classification?"

❌ **Too broad:** "What is AI?"  
✅ **Focused:** "How does this paper apply AI to medical diagnosis?"

### 3. Improving Answer Quality

- **Lower temperature (0.1-0.3)**: More factual, focused answers
- **Higher top_k (5-10)**: More context for complex questions
- **Stricter threshold (0.5-0.7)**: Only highly relevant chunks

### 4. Managing Large Collections

```python
# Use multiple collections for different topics
ml_papers = get_collection(client, "ml_papers")
nlp_papers = get_collection(client, "nlp_papers")

# Query specific collection
result = answer_question_with_rag(question, ml_papers)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **PyMuPDF** - PDF text extraction
- **Ollama** - Local LLM inference
- **ChromaDB** - Vector database
- **Qwen Team** - Qwen3 language model
- **Nomic AI** - nomic-embed-text embedding model

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for researchers who value privacy and local AI**
