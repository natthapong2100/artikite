"""
config.py
─────────
Central configuration for the Art History RAG Agentic System.
Change settings here — everything else reads from this file.
"""

# ── Ollama ──────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_LLM_MODEL  = "llama3.1"          # For text generation
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # For embeddings (pull with: ollama pull nomic-embed-text)
                                          # Fallback: "llama3.1" can also do embeddings

# ── ChromaDB ────────────────────────────────────────────
CHROMA_PERSIST_DIR     = "./chroma_db"   # Where ChromaDB stores vectors on disk
CHROMA_COLLECTION_NAME = "art_history"

# ── RAG ─────────────────────────────────────────────────
RAG_TOP_K           = 3     # How many chunks to retrieve per query
CHUNK_SIZE          = 400   # Characters per document chunk
CHUNK_OVERLAP       = 50    # Overlap between chunks to preserve context

# ── LLM Generation ──────────────────────────────────────
LLM_TEMPERATURE     = 0.7
LLM_MAX_TOKENS      = 1024

# ── LangGraph ───────────────────────────────────────────
MAX_REVISION_LOOPS  = 2     # Max times the validator can send work back

# ── CrewAI ──────────────────────────────────────────────
CREW_VERBOSE        = True
