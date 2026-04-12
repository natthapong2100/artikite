# Artikite — Art History RAG Agentic System

## Stack
- **LLM**: Ollama local (`qwen3.5`) — no external API keys needed
- **Guardrail LLM**: Ollama local (`qwen3.5:0.8b`) — lightweight, safety checks only
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector DB**: ChromaDB (persisted at `./chroma_db`)
- **Orchestration**: LangGraph (stateful graph) + CrewAI (multi-agent)
- **Museum data**: MET Museum via MCP protocol (server spawned automatically)
- **MCP**: `mcp` package — both server (`FastMCP`) and client (`ClientSession`)

## File Map

```
artikite/
├── main.py                    # Entry point: ingest → MCP → validate query (safety) → LangGraph → CrewAI → output check → save
├── config.py                  # All config constants (URLs, model names, RAG params)
├── museum_mcp_server.py       # MCP server: 2 tools (search_met_artworks, get_met_artwork)
│
├── agents/
│   ├── guardrails.py          # Input + output guardrails (validate_query, validate_output)
│   ├── mcp_client.py          # Sync MCP client — spawns museum_mcp_server.py, calls tools
│   ├── langgraph_flow.py      # LangGraph: Planner→Researcher→Writer→Validator→[Reviser]
│   ├── crew.py                # CrewAI: Research Specialist→Art Critic→Museum Curator
│   ├── tools.py               # LangChain Tool wrappers for ChromaDB RAG
│   └── museum_tools.py        # LangChain Tool wrappers — METMuseumSearch + METGetArtwork
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
4. **Safety check** — `validate_query(query)` via `qwen3.5:0.8b`; blocks UNSAFE queries only
5. **STEP 4** — `run_langgraph_research(query)` → returns `{final_essay, log, ...}`
6. **STEP 5** — `run_crew_analysis(query)` → returns curated essay string
7. **STEP 6** — `validate_output(essay, query)` — heuristic + LLM check on final essay
8. **STEP 7** — Combined report saved to `reports/report_<query>_<timestamp>.md`

## MCP Architecture

```
main.py
  ├─ mcp_client.connect()          → spawns museum_mcp_server.py subprocess
  ├─ LangGraph researcher_node
  │     ├─ met_search_tool()
  │     │     └─ mcp_client.call_tool("search_met_artworks", {...})
  │     │           └─ [MCP stdio] → museum_mcp_server.py → MET REST API
  │     └─ met_get_artwork_tool()   ← auto-called for top result ID
  │           └─ mcp_client.call_tool("get_met_artwork", {...})
  │                 └─ [MCP stdio] → museum_mcp_server.py → MET REST API
  └─ CrewAI Research Specialist
        ├─ met_search_tool()
        │     └─ mcp_client.call_tool("search_met_artworks", {...})
        └─ met_get_artwork_tool()   ← agent can call explicitly with object ID
              └─ mcp_client.call_tool("get_met_artwork", {...})
```

### `agents/mcp_client.py`
- Spawns `museum_mcp_server.py` as a stdio subprocess on `connect()`
- Background asyncio thread loop (mcp client is async-only; sync wrapper via `asyncio.run_coroutine_threadsafe`)
- `_stdio_cm` kept as module-level ref to prevent subprocess GC-kill
- `connect()` — idempotent; `disconnect()` — graceful async teardown of session + subprocess
- `call_tool(tool_name, arguments)` — sync, auto-connects, **reconnects once** on session drop
- Uses raw `mcp` package: `ClientSession`, `StdioServerParameters`, `stdio_client`

### `museum_mcp_server.py`
- Exposes 2 MCP tools: `search_met_artworks(query, max_results)`, `get_met_artwork(object_id)`
- Self-contained — defines `_HEADERS` locally, does NOT import from `museum_tools.py`
- Can still be run standalone: `python museum_mcp_server.py` (for external MCP clients)

## Guardrails (`agents/guardrails.py`)

Uses `qwen3.5:0.8b` (`GUARDRAIL_MODEL`) at `temperature=0.0`, `num_predict=200`.

### Input guardrail — `validate_query(query)`
- Length check: rejects queries under 10 chars (no LLM call)
- LLM binary check: asks "is this UNSAFE?" — only blocks harmful content / prompt injection
- Off-topic queries pass through (pipeline handles them)
- `_strip_think()` handles both complete and truncated `<think>` blocks

### Output guardrail — `validate_output(essay, query)`
- **Layer 1 — heuristic** (no LLM, fast):
  - Rejects essay under 200 chars
  - Rejects if raw `<think>` tokens are present in the essay
  - Rejects if essay starts with any refusal phrase ("i cannot", "i'm sorry", etc.)
- **Layer 2 — LLM relevance check**:
  - Sends first 500 chars of essay + original query to `GUARDRAIL_MODEL`
  - Expects YES/NO — blocks if essay doesn't answer the question

## LangGraph Nodes (`agents/langgraph_flow.py`)
- `strip_think(text)` — shared helper; strips complete + truncated `<think>` blocks
- `planner_node` — generates 3 research questions; output wrapped with `strip_think()`
- `researcher_node` — routes to ChromaDB or MET API via `_route_sources()`; after MET search, auto-fetches full detail for top result ID via `get_met_artwork`
- `writer_node` — drafts essay from research; output wrapped with `strip_think()`
- `validator_node` — heuristic pre-checks (length, required sections) then LLM review; LLM output wrapped with `strip_think()`
- `reviser_node` — rewrites if needed (controlled by `config.MAX_REVISION_LOOPS`); output wrapped with `strip_think()`

### Validator pre-checks (heuristic, no LLM)
- Rejects draft under 300 chars
- Checks all 4 required sections present (case-insensitive): `## introduction`, `## key findings`, `## historical significance`, `## conclusion`
- Fails fast with `NEEDS_IMPROVEMENT` before LLM review

### Source Routing (`_route_sources`)
- LLM classifies query → `chromadb`, `met`, or both
- qwen3.5 `<think>...</think>` tokens stripped before parsing

## CrewAI Agents (`agents/crew.py`)
| Agent | Tools | Role |
|-------|-------|------|
| Research Specialist | ArtHistorySearch, ArtistSearch, MovementSearch, METMuseumSearch, METGetArtwork | Gathers facts |
| Art Critic | ArtComparison | Cultural analysis essay |
| Museum Curator | ArtHistorySearch | Fact-checks, polishes, adds Curator's Note |

### CrewAI Routing (`_detect_sources` + `_build_tool_instructions`)
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
| METGetArtwork | agents/museum_tools.py | → mcp_client → MCP → MET REST API |

> **LangChain tools**: `Tool(name=..., func=..., description=...)`
> **CrewAI tools**: `@crewai_tool("ToolName")` decorator wrapping the same functions

## Key Config (`config.py`)
```python
OLLAMA_LLM_MODEL   = "qwen3.5"
GUARDRAIL_MODEL    = "qwen3.5:0.8b"     # lightweight, safety checks only
OLLAMA_EMBED_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = "./chroma_db"
RAG_TOP_K          = 2
MAX_REVISION_LOOPS = 0                   # 0 = validator never sends back for revision
CREW_VERBOSE       = True
MET_API_BASE       = "https://collectionapi.metmuseum.org/public/collection/v1"
MCP_TOOL_TIMEOUT   = 60                  # seconds per MCP tool call
```

## Run
```bash
ollama pull qwen3.5
ollama pull qwen3.5:0.8b
ollama pull nomic-embed-text
python main.py               # full pipeline (auto-starts MCP server internally)
python museum_mcp_server.py  # standalone MCP server for external clients only
```

## Important Notes
- MET API requires `User-Agent` header or returns 403 — `_HEADERS` defined in `museum_mcp_server.py`
- qwen3.5 emits `<think>...</think>` reasoning blocks — always strip before parsing; use `strip_think()` (langgraph_flow) or `_strip_think()` (guardrails); both handle truncated blocks (no closing tag)
- CrewAI tools must use `@crewai_tool` decorator, NOT bare `Tool()` objects
- MOMA was removed — only MET + ChromaDB are active data sources
- `museum_tools.py` no longer uses httpx — all MET calls go through `mcp_client.call_tool()`
- `mcp_client.disconnect()` is called in a `finally` block in `main.py` — always runs even on pipeline error
- Guardrail model (`qwen3.5:0.8b`) is separate from pipeline LLM (`qwen3.5`) — must be pulled separately
