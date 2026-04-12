# ── Ollama ──────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_LLM_MODEL  = "qwen3.5"           # For text generation
GUARDRAIL_MODEL   = "qwen3.5:0.8b"     # Lightweight model for safety checks (fast)
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # For embeddings (pull with: ollama pull nomic-embed-text)
                                          # Fallback: "llama3.1" can also do embeddings

# ── ChromaDB ────────────────────────────────────────────
CHROMA_PERSIST_DIR     = "./chroma_db"   # Where ChromaDB stores vectors on disk
CHROMA_COLLECTION_NAME = "art_history"

# ── RAG ─────────────────────────────────────────────────
RAG_TOP_K           = 2     # How many chunks to retrieve per query
CHUNK_SIZE          = 400   # Characters per document chunk
CHUNK_OVERLAP       = 50    # Overlap between chunks to preserve context

# ── LLM Generation ──────────────────────────────────────
LLM_TEMPERATURE     = 0.7
LLM_MAX_TOKENS      = 1024

# ── LangGraph ───────────────────────────────────────────
MAX_REVISION_LOOPS  = 0     # Max times the validator can send work back, but is SKIP!

# ── CrewAI ──────────────────────────────────────────────
CREW_VERBOSE        = True # true is to show the thinking process of the agents, false is to only show final output

# ── Museum APIs ──────────────────────────────────────────
MET_API_BASE        = "https://collectionapi.metmuseum.org/public/collection/v1"
MCP_TOOL_TIMEOUT    = 60   # seconds to wait for each MCP tool call
