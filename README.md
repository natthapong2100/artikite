# 🎨 Art History RAG Agentic System

A production-style agentic AI project combining **RAG (Retrieval-Augmented Generation)** with multi-agent orchestration — all running locally with Ollama.

---

## 📁 Project Structure

```
art_history_rag/
│
├── main.py                    ← Entry point. Run this.
├── config.py                  ← All settings in one place
├── requirements.txt
│
├── data/
│   └── art_knowledge.py       ← Art history corpus (14 documents)
│                                 Artists: da Vinci, Michelangelo, Raphael,
│                                          Caravaggio, Rembrandt, Monet,
│                                          Van Gogh, Picasso, Kahlo
│                                 Movements: Renaissance, Baroque,
│                                            Impressionism, Cubism, Surrealism
│
├── rag/
│   ├── embedder.py            ← Ollama embeddings wrapper
│   └── vector_store.py        ← ChromaDB: ingest, chunk, retrieve
│
└── agents/
    ├── tools.py               ← LangChain Tools (shared by LangGraph + CrewAI)
    ├── langgraph_flow.py      ← LangGraph stateful workflow
    └── crew.py                ← CrewAI multi-agent crew
```

### Why this structure?

| File | Responsibility | Could swap with... |
|------|---------------|-------------------|
| `config.py` | Central settings | Environment variables / .env |
| `data/art_knowledge.py` | Knowledge corpus | PDF loader, Wikipedia API, web scraper |
| `rag/embedder.py` | Text → vectors | sentence-transformers, OpenAI embeddings |
| `rag/vector_store.py` | Store + retrieve | Pinecone, Weaviate, FAISS |
| `agents/tools.py` | LLM-callable functions | Any Python functions |
| `agents/langgraph_flow.py` | Workflow orchestration | Simpler LangChain chains |
| `agents/crew.py` | Multi-agent team | AutoGen, single agent |

---

## 🏗️ Architecture

```
USER QUERY
    │
    ▼
main.py
    │
    ├── [Phase 1] LangGraph Workflow
    │       Planner Node     → Breaks query into research plan
    │       Researcher Node  → Calls RAG tools (ChromaDB)
    │       Writer Node      → Synthesizes essay from research
    │       Validator Node   → Quality check → conditional routing
    │       Reviser Node     → Improves draft if needed (loops back)
    │
    └── [Phase 2] CrewAI Crew
            ResearchSpecialist → Gathers facts via RAG tools
            ArtCritic          → Writes cultural analysis
            MuseumCurator      → Reviews, fact-checks, polishes

Both phases use the same LangChain Tools → same ChromaDB → same Ollama
```

---

## 🔑 Key Concepts

### RAG (Retrieval-Augmented Generation)
Instead of relying solely on the LLM's training knowledge (which may be outdated or vague), RAG:
1. **Stores** your documents as vectors in ChromaDB
2. **Retrieves** the most relevant chunks for each query
3. **Augments** the LLM prompt with those chunks
4. **Generates** a grounded, accurate answer

```
Query → Embed query → ChromaDB similarity search → Top-K chunks → LLM prompt → Answer
```

### LangChain (Foundation Layer)
- `Ollama` — wraps Ollama as a LangChain LLM
- `PromptTemplate` + `LLMChain` — structured prompt pipelines
- `Tool` — wraps Python functions so agents can call them
- Used by both LangGraph and CrewAI as the shared base

### LangGraph (Workflow Orchestration)
- `StateGraph` — graph where nodes are functions, edges are transitions
- Typed shared state (`TypedDict`) passed between nodes
- **Conditional edges** — route to different nodes based on logic
- **Cycles** — loop back (revise → validate → revise) until quality passes

### CrewAI (Multi-Agent Collaboration)
- `Agent` — role + goal + backstory + tools = personality and capability
- `Task` — description + expected output + `context` (task dependencies)
- `Crew` — assembles agents + tasks + process type
- `Process.sequential` — tasks run in order, each sees previous outputs
- `allow_delegation=True` — curator can delegate back to critic

### ChromaDB (Vector Database)
- Persistent storage for document embeddings
- Cosine similarity search for semantic retrieval
- Metadata filtering (e.g., only search 'artist' documents)
- `get_or_create_collection` — idempotent, safe to call multiple times

---

## 🚀 Setup & Run

### 1. Install Ollama and pull models
```bash
# Install Ollama (Mac/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1           # LLM for generation
ollama pull nomic-embed-text   # Embedding model (faster & better for RAG)

# Start server
ollama serve
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python main.py
```

You'll see an interactive menu. Choose a sample query or type your own.

### 4. Change the embedding model (optional)
In `config.py`:
```python
OLLAMA_EMBED_MODEL = "nomic-embed-text"   # Recommended
# OLLAMA_EMBED_MODEL = "llama3.1"         # Fallback if nomic not available
```

---

## 💡 Interview Talking Points

**"Walk me through your RAG implementation."**
> "Documents are chunked into 400-character pieces with 50-character overlap to preserve context. Each chunk is embedded using Ollama's nomic-embed-text model and stored in ChromaDB with metadata. At query time, I embed the query with the same model, do cosine similarity search, retrieve the top-K chunks, and inject them into the LLM prompt. This grounds the LLM in actual knowledge rather than hallucination."

**"Why use LangGraph instead of a simple LangChain chain?"**
> "LangChain chains are linear — great for simple pipelines. LangGraph gives you explicit state management, conditional routing, and cycles. In this project, after writing the essay, the validator might send it back to the reviser. A simple chain can't loop; LangGraph handles it naturally."

**"What's the difference between LangGraph and CrewAI?"**
> "LangGraph gives you precise control over the flow — you define exactly which node runs when. CrewAI gives you role-based agents that collaborate more autonomously — you define who each agent is and what they should do, and CrewAI manages the handoffs. LangGraph is like a flowchart; CrewAI is like a team."

**"Why ChromaDB over other vector databases?"**
> "ChromaDB is lightweight, runs locally with no external services, has a simple Python API, and persists to disk. For production I'd consider Pinecone for scale or Weaviate for hybrid search, but for local development ChromaDB is ideal."

**"How would you scale this to production?"**
> "Replace the mock corpus with a real document ingestion pipeline (PDFs, Wikipedia, museum APIs). Use a proper embedding model like text-embedding-3-small. Add LangGraph's MemorySaver for conversation persistence. Deploy the LLM behind vLLM or TensorRT-LLM for low-latency serving. Containerize with Docker, orchestrate with Kubernetes."

---

## 📊 Sample Output

```
═══════════════════════════════════════════════════
  STEP 3: LangGraph Research Workflow
  Running: Planner → Researcher (RAG) → Writer → Validator → [Reviser]

  🗺️  [LangGraph] PLANNER node running...
  🔍  [LangGraph] RESEARCHER node running...
       → Calling ArtHistorySearch tool (RAG)...
       → Calling ArtistSearch tool (RAG, artist filter)...
       → Calling MovementSearch tool (RAG, movement filter)...
  ✍️   [LangGraph] WRITER node running...
  ✅  [LangGraph] VALIDATOR node running...
       Validation: VALID

═══════════════════════════════════════════════════
  STEP 4: CrewAI Multi-Agent Analysis
  Running: Research Specialist → Art Critic → Museum Curator
```

---

## 🔧 Extending the Project

- **Add more documents**: Extend `data/art_knowledge.py` or add a PDF loader
- **Real web search**: Replace the corpus with live Wikipedia/museum API calls
- **Image analysis**: Add a vision model (LLaVA via Ollama) to analyze artwork images
- **Conversation memory**: Add LangGraph `MemorySaver` for multi-turn conversations
- **API endpoint**: Wrap `main.py` in FastAPI for a REST interface
- **Streaming**: Use `langgraph_app.astream()` for real-time token streaming
