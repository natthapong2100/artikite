"""
LangChain tool for MET Museum search.

Uses MCP as the transport layer:
    met_search() → mcp_client.call_tool() → museum_mcp_server.py → MET REST API
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import Tool
from agents import mcp_client


# ── Core function ─────────────────────────────────────────────────────────────

def met_search(query: str) -> str:
    """Search the Metropolitan Museum of Art collection via MCP."""
    return mcp_client.call_tool("search_met_artworks", {"query": query})


# ── LangChain Tool object ─────────────────────────────────────────────────────

met_search_tool = Tool(
    name="METMuseumSearch",
    func=met_search,
    description=(
        "Search the Metropolitan Museum of Art collection for real artworks, artists, "
        "and periods. Returns live data via MCP from MET's public API including titles, "
        "dates, mediums, and direct URLs. Input: artist name, artwork title, or art term."
    ),
)
