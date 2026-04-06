# Artikite — Art History RAG Agentic System

## Stack
- **LLM**: Ollama local (`qwen3.5`) — no external API keys needed
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector DB**: ChromaDB (persisted at `./chroma_db`)
- **Orchestration**: LangGraph (stateful graph) + CrewAI (multi-agent)
- **Museum data**: MET Museum via MCP protocol (server spawned automatically)
- **MCP**: `mcp` package — both server (`FastMCP`) and client (`ClientSession`)

## File Map

```
artikite/
├── main.py                    # Entry point: ingest → MCP connect → LangGraph → CrewAI → save
├── config.py                  # All config constants (URLs, model names, RAG params)
├── museum_mcp_server.py       # MCP server: 2 tools (search_met_artworks, get_met_artwork)
│
├── agents/
│   ├── mcp_client.py          # Sync MCP client — spawns museum_mcp_server.py, calls tools
│   ├── langgraph_flow.py      # LangGraph: Planner→Researcher→Writer→Validator→[Reviser]
│   ├── crew.py                # CrewAI: Research Specialist→Art Critic→Museum Curator
│   ├── tools.py               # LangChain Tool wrappers for ChromaDB RAG
│   └── museum_tools.py        # LangChain Tool wrapper — calls MET via mcp_client
│
├── rag/
│   ├── vector_store.py        # ChromaDB ingest + query
│   └── embedder.py            # Ollama nomic-embed-text embeddings
│
└── data/
    └── art_knowledge.py       # Static art history corpus (ingested into ChromaDB)
```

## Pipeline Flow (`main.py`)
1. **STEP 1** — Ingest `data/art_knowledge.py` into ChromaDB (idempotent)
2. **STEP 2** — `mcp_client.connect()` spawns `museum_mcp_server.py` subprocess, opens MCP session
3. **STEP 3** — User selects query
4. **STEP 4** — `run_langgraph_research(query)` → returns `{final_essay, log, ...}`
5. **STEP 5** — `run_crew_analysis(query)` → returns curated essay string
6. **STEP 6** — Combined report saved to `reports/report_<query>_<timestamp>.md`

## MCP Architecture

```
main.py
  ├─ mcp_client.connect()          → spawns museum_mcp_server.py subprocess
  ├─ LangGraph researcher_node
  │     └─ met_search_tool()
  │           └─ mcp_client.call_tool("search_met_artworks", {...})
  │                 └─ [MCP stdio] → museum_mcp_server.py → MET REST API
  └─ CrewAI Research Specialist
        └─ met_search_tool()
              └─ mcp_client.call_tool("search_met_artworks", {...})
                    └─ [MCP stdio] → museum_mcp_server.py → MET REST API
```

### `agents/mcp_client.py`
- Spawns `museum_mcp_server.py` as a stdio subprocess on `connect()`
- Background asyncio thread loop (mcp client is async-only; sync wrapper via `asyncio.run_coroutine_threadsafe`)
- `_stdio_cm` kept as module-level ref to prevent subprocess GC-kill
- `call_tool(tool_name, arguments)` — sync, auto-connects on first call
- Uses raw `mcp` package: `ClientSession`, `StdioServerParameters`, `stdio_client`

### `museum_mcp_server.py`
- Exposes 2 MCP tools: `search_met_artworks(query, max_results)`, `get_met_artwork(object_id)`
- Self-contained — defines `_HEADERS` locally, does NOT import from `museum_tools.py`
- Can still be run standalone: `python museum_mcp_server.py` (for external MCP clients)

## LangGraph Nodes (`agents/langgraph_flow.py`)
- `planner_node` — generates 3 research questions
- `researcher_node` — routes to ChromaDB or MET API via `_route_sources()`
- `writer_node` — drafts essay from research
- `validator_node` — scores draft (VALID / NEEDS_IMPROVEMENT)
- `reviser_node` — rewrites if needed (controlled by `config.MAX_REVISION_LOOPS`)

### Source Routing (`_route_sources`)
- Keyword fast-path: "met"/"metropolitan" in query → forces `["met"]`
- Otherwise: LLM classifies query → `chromadb`, `met`, or both
- qwen3.5 `<think>...</think>` tokens stripped with regex before parsing

## CrewAI Agents (`agents/crew.py`)
| Agent | Tools | Role |
|-------|-------|------|
| Research Specialist | ArtHistorySearch, ArtistSearch, MovementSearch, METMuseumSearch | Gathers facts |
| Art Critic | ArtComparison | Cultural analysis essay |
| Museum Curator | ArtHistorySearch | Fact-checks, polishes, adds Curator's Note |

### CrewAI Routing (`_detect_sources` + `_build_tool_instructions`)
- Same keyword logic as LangGraph
- MET query → task instructions say "Use METMuseumSearch"
- General query → task instructions say "Use ArtHistorySearch, ArtistSearch, MovementSearch"

## Tools
| Name | File | Backend |
|------|------|---------|
| ArtHistorySearch | agents/tools.py | ChromaDB semantic search |
| ArtistSearch | agents/tools.py | ChromaDB semantic search |
| MovementSearch | agents/tools.py | ChromaDB semantic search |
| ArtComparison | agents/tools.py | ChromaDB semantic search |
| METMuseumSearch | agents/museum_tools.py | → mcp_client → MCP → MET REST API |

> **LangChain tools**: `Tool(name=..., func=..., description=...)`
> **CrewAI tools**: `@crewai_tool("ToolName")` decorator wrapping the same functions

## Key Config (`config.py`)
```python
OLLAMA_LLM_MODEL   = "qwen3.5"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = "./chroma_db"
RAG_TOP_K          = 2
MAX_REVISION_LOOPS = 0        # 0 = validator never sends back for revision
CREW_VERBOSE       = True
MET_API_BASE       = "https://collectionapi.metmuseum.org/public/collection/v1"
```

## Run
```bash
python main.py               # full pipeline (auto-starts MCP server internally)
python museum_mcp_server.py  # standalone MCP server for external clients only
```

## Important Notes
- MET API requires `User-Agent` header or returns 403 — `_HEADERS` defined in `museum_mcp_server.py`
- qwen3.5 emits `<think>...</think>` reasoning blocks — always strip with `re.sub` before parsing LLM output
- CrewAI tools must use `@crewai_tool` decorator, NOT bare `Tool()` objects
- MOMA was removed — only MET + ChromaDB are active data sources
- `museum_tools.py` no longer uses httpx — all MET calls go through `mcp_client.call_tool()`
