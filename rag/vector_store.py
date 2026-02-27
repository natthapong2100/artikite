"""
rag/vector_store.py
────────────────────
ChromaDB setup, document ingestion, and semantic retrieval.

Responsibilities:
    1. Create and persist a ChromaDB collection
    2. Ingest art history documents (chunking + embedding + storing)
    3. Retrieve relevant chunks for a given query (semantic search)

Why chunk documents?
    LLMs have a context window limit. A full article might be 2000+ tokens.
    By splitting into smaller chunks (e.g., 400 characters), we only feed
    the most relevant parts to the LLM — keeping responses focused and fast.

ChromaDB concepts:
    Collection  = like a table in a SQL database
    Document    = the raw text chunk
    Embedding   = the vector representation of the chunk
    Metadata    = extra info stored alongside (title, category, chunk index)
    ID          = unique string identifier for each chunk
"""

import chromadb
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.embedder import get_embedding, get_embeddings_batch
from data.art_knowledge import get_all_documents


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks.

    Args:
        text:       The full document text.
        chunk_size: Max characters per chunk.
        overlap:    Characters shared between adjacent chunks (preserves context).

    Returns:
        List of text chunks.

    Example:
        chunk_text("ABCDE", chunk_size=3, overlap=1) → ["ABC", "CDE"]
    """
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # Move forward, but keep 'overlap' chars
    return chunks


# ── ChromaDB Client ──────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.Client:
    """
    Return a persistent ChromaDB client.
    Data is saved to CHROMA_PERSIST_DIR so you don't re-embed on every run.
    """
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_or_create_collection(client: chromadb.Client) -> chromadb.Collection:
    """
    Get existing collection or create a new one.
    We use embedding_function=None because we supply our own Ollama embeddings.
    """
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=None,   # We handle embeddings ourselves via Ollama
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )


# ── Ingestion ────────────────────────────────────────────────────────────────

def ingest_documents(force_reingest: bool = False) -> chromadb.Collection:
    """
    Load art history documents → chunk → embed → store in ChromaDB.

    Args:
        force_reingest: If True, delete and recreate the collection even if it exists.

    Returns:
        The populated ChromaDB collection.
    """
    client = get_chroma_client()

    # Check if already ingested
    existing = client.list_collections()
    collection_names = [c.name for c in existing]

    if config.CHROMA_COLLECTION_NAME in collection_names and not force_reingest:
        collection = get_or_create_collection(client)
        count = collection.count()
        if count > 0:
            print(f"  ✅ ChromaDB: Collection '{config.CHROMA_COLLECTION_NAME}' "
                  f"already exists with {count} chunks. Skipping ingestion.")
            return collection

    # Delete existing if force re-ingest
    if config.CHROMA_COLLECTION_NAME in collection_names and force_reingest:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
        print("  🗑️  Deleted existing collection for re-ingestion.")

    collection = get_or_create_collection(client)
    documents = get_all_documents()

    print(f"\n  📚 Ingesting {len(documents)} art history documents into ChromaDB...")

    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_text(doc["content"])
        print(f"     '{doc['title']}' → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "doc_id":   doc["id"],
                "title":    doc["title"],
                "category": doc["category"],
                "chunk":    i
            })

    # Embed all chunks
    print(f"\n  🔢 Embedding {len(all_chunks)} chunks with Ollama ({config.OLLAMA_EMBED_MODEL})...")
    all_embeddings = get_embeddings_batch(all_chunks)

    collection.add(
        documents=all_chunks,       # Raw text chunks (for retrieval)
        embeddings=all_embeddings,  # Their vector representations
        ids=all_ids,
        metadatas=all_metadatas
    )

    print(f"\n  ✅ ChromaDB ingestion complete: {collection.count()} chunks stored.")
    return collection


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, collection: chromadb.Collection = None,
             top_k: int = config.RAG_TOP_K,
             category_filter: str = None) -> list[dict]:
    """
    Semantic search: find the most relevant art history chunks for a query.

    Args:
        query:           The user's question or search phrase.
        collection:      ChromaDB collection (auto-loaded if None).
        top_k:           Number of results to return.
        category_filter: Optional filter: 'artist' or 'movement'.

    Returns:
        List of dicts with keys: text, title, category, distance, doc_id
    """
    if collection is None:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

    # Embed the query using the same model as the documents
    query_embedding = get_embedding(query)

    # Build optional metadata filter
    where_filter = {"category": category_filter} if category_filter else None

    # Query ChromaDB for nearest vectors (cosine similarity)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Format results for easy use
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "text":     results["documents"][0][i],
            "title":    results["metadatas"][0][i]["title"],
            "category": results["metadatas"][0][i]["category"],
            "doc_id":   results["metadatas"][0][i]["doc_id"],
            "distance": results["distances"][0][i],   # Lower = more similar
        })

    return formatted


def format_retrieved_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context string for the LLM prompt.
    Groups chunks by source document to avoid repetition.
    """
    if not chunks:
        return "No relevant information found in the knowledge base."

    context_parts = []
    seen_docs = set()

    for chunk in chunks:
        doc_key = chunk["doc_id"]
        header = f"[Source: {chunk['title']} ({chunk['category']})]"
        if doc_key not in seen_docs:
            context_parts.append(f"{header}\n{chunk['text']}")
            seen_docs.add(doc_key)
        else:
            context_parts.append(chunk['text'])

    return "\n\n---\n\n".join(context_parts)


# ── Test when run directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ChromaDB vector store...")
    collection = ingest_documents()
    print("\nTesting retrieval: 'Who painted the Sistine Chapel?'")
    results = retrieve("Who painted the Sistine Chapel?", collection)
    for r in results:
        print(f"\n  📄 {r['title']} (distance: {r['distance']:.3f})")
        print(f"     {r['text'][:150]}...")
