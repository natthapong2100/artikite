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

