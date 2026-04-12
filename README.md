# 🎨 Artikite — Art History RAG Agentic System

A production-style agentic AI project combining **RAG**, **multi-agent orchestration**, **MCP tool transport**, and **guardrails** — all running locally with Ollama, no external API keys required.

---

## 📁 Project Structure

```
artikite/
├── main.py                    ← Entry point. Run this.
├── config.py                  ← All settings in one place
├── museum_mcp_server.py       ← MCP server exposing MET Museum tools
├── requirements.txt
│
├── agents/
│   ├── guardrails.py          ← Input + output safety guardrails
│   ├── mcp_client.py          ← Sync MCP client (spawns museum server subprocess)
│   ├── langgraph_flow.py      ← LangGraph stateful workflow
│   ├── crew.py                ← CrewAI multi-agent crew
│   ├── tools.py               ← LangChain Tools for ChromaDB RAG
│   └── museum_tools.py        ← LangChain Tools for MET Museum via MCP
│
├── rag/
│   ├── embedder.py            ← Ollama embeddings wrapper
│   └── vector_store.py        ← ChromaDB: ingest, chunk, retrieve
│
└── data/
    └── art_knowledge.py       ← Art history corpus (14 documents)
                                  Artists: da Vinci, Michelangelo, Raphael,
                                           Caravaggio, Rembrandt, Monet,
                                           Van Gogh, Picasso, Kahlo
                                  Movements: Renaissance, Baroque,
                                             Impressionism, Cubism, Surrealism
```

### Why this structure?

| File | Responsibility | Could swap with... |
|------|---------------|-------------------|
| `config.py` | Central settings | Environment variables / .env |
| `data/art_knowledge.py` | Knowledge corpus | PDF loader, Wikipedia API, web scraper |
| `rag/embedder.py` | Text → vectors | sentence-transformers, OpenAI embeddings |
| `rag/vector_store.py` | Store + retrieve | Pinecone, Weaviate, FAISS |
| `agents/guardrails.py` | Safety checks | Dedicated safety APIs |
| `agents/mcp_client.py` | MCP tool transport | Direct HTTP calls |
| `museum_mcp_server.py` | MET Museum API | Any museum / data API |
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
    ├── [STEP 1] ChromaDB ingest (idempotent)
    │
    ├── [STEP 2] MCP connect → spawns museum_mcp_server.py subprocess
    │
    ├── [STEP 3] User selects query
    │
    ├── [Safety] validate_query() — blocks UNSAFE queries (qwen3.5:0.8b)
    │
    ├── [STEP 4] LangGraph Workflow
    │       Planner Node     → Breaks query into 3 research questions
    │       Researcher Node  → Routes to ChromaDB and/or MET via MCP
    │                          (auto-fetches full artwork detail for top MET result)
    │       Writer Node      → Synthesizes essay with required sections
    │       Validator Node   → Heuristic pre-checks + LLM quality review
    │       Reviser Node     → Improves draft if needed (loops back)
    │
    ├── [STEP 5] CrewAI Crew
    │       ResearchSpecialist → Gathers facts via RAG + MET tools
    │       ArtCritic          → Writes cultural analysis
    │       MuseumCurator      → Reviews, fact-checks, adds Curator's Note
    │
    ├── [STEP 6] validate_output() — heuristic + LLM check on final essay
    │
    └── [STEP 7] Save combined report → reports/report_<query>_<timestamp>.md
```

---

## 🔌 MCP (Model Context Protocol)

The MET Museum API is exposed as an MCP server and consumed via a sync MCP client — making the tool transport pluggable and protocol-standard.

```
met_search_tool() / met_get_artwork_tool()
    └─ mcp_client.call_tool(...)
          └─ [MCP stdio transport]
                └─ museum_mcp_server.py
                      └─ MET REST API  (collectionapi.metmuseum.org)
```

**Two MET tools exposed over MCP:**

| Tool | Description |
|------|-------------|
| `search_met_artworks(query, max_results)` | Search MET collection, returns titles, artists, dates, URLs |
| `get_met_artwork(object_id)` | Full detail for one artwork: bio, medium, dimensions, gallery, tags |

**MCP client features:**
- Auto-spawns `museum_mcp_server.py` as a subprocess on first connect
- Background asyncio event loop — sync wrapper for use in synchronous code
- Graceful `disconnect()` — properly closes session and subprocess
- Auto-reconnects once if session drops mid-run

---

## 🛡️ Guardrails (`agents/guardrails.py`)

Uses `qwen3.5:0.8b` — a smaller, faster model dedicated to safety checks only.

### Input guardrail — runs before the pipeline
- Hard rejects queries under 10 characters
- LLM binary check: is this query **UNSAFE**? (harmful content, prompt injection, instruction override)
- Off-topic queries pass through — the pipeline handles them

### Output guardrail — runs after the pipeline, before saving
| Check | Type | Rejects if... |
|-------|------|---------------|
| Length | Heuristic | Essay under 200 characters |
| Think tokens | Heuristic | Raw `<think>` appears in essay |
| Refusal phrases | Heuristic | Essay starts with "I cannot", "I'm sorry", "As an AI", etc. |
| Relevance | LLM | Essay doesn't actually answer the original question |

---

## 🔑 Key Concepts

### RAG (Retrieval-Augmented Generation)
Instead of relying solely on the LLM's training knowledge, RAG:
1. **Stores** documents as vectors in ChromaDB
2. **Retrieves** the most relevant chunks for each query
3. **Augments** the LLM prompt with those chunks
4. **Generates** a grounded, accurate answer

```
Query → Embed → ChromaDB similarity search → Top-K chunks → LLM prompt → Answer
```

### LangGraph (Workflow Orchestration)
- `StateGraph` — graph where nodes are functions, edges are transitions
- Typed shared state (`TypedDict`) passed between nodes
- **Conditional edges** — route to different nodes based on validation result
- **Cycles** — loop back (revise → validate → revise) until quality passes or loop limit reached

### CrewAI (Multi-Agent Collaboration)
- `Agent` — role + goal + backstory + tools = personality and capability
- `Task` — description + expected output + `context` (task dependencies)
- `Crew` — assembles agents + tasks + process type
- `Process.sequential` — tasks run in order, each sees previous outputs

### ChromaDB (Vector Database)
- Persistent storage for document embeddings
- Cosine similarity search for semantic retrieval
- `get_or_create_collection` — idempotent, safe to call multiple times

---

## 📊 Pipeline Step Summary

| Step | Component | Description |
|------|-----------|-------------|
| 1 | ChromaDB | Ingest art history corpus (idempotent) |
| 2 | MCP Client | Spawn MET museum server subprocess |
| 3 | Input | User selects or types query |
| Safety | Guardrail | Block unsafe queries via `qwen3.5:0.8b` |
| 4 | LangGraph | Plan → Research → Write → Validate → [Revise] |
| 5 | CrewAI | Research → Critique → Curate |
| 6 | Guardrail | Validate output quality before saving |
| 7 | Report | Save combined markdown report |
