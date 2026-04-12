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


def met_get_artwork(object_id: str) -> str:
    """Get full details for a specific MET artwork by its object ID via MCP."""
    try:
        oid = int(str(object_id).strip())
    except ValueError:
        return f"[MCP] Invalid object ID: {object_id}"
    return mcp_client.call_tool("get_met_artwork", {"object_id": oid})


# ── LangChain Tool objects ────────────────────────────────────────────────────

met_search_tool = Tool(
    name="METMuseumSearch",
    func=met_search,
    description=(
        "Search the Metropolitan Museum of Art collection for real artworks, artists, "
        "and periods. Returns live data via MCP from MET's public API including titles, "
        "dates, mediums, and direct URLs. Input: artist name, artwork title, or art term."
    ),
)

met_get_artwork_tool = Tool(
    name="METGetArtwork",
    func=met_get_artwork,
    description=(
        "Get full details for a specific MET Museum artwork by its object ID. "
        "Returns artist biography, medium, dimensions, department, credit line, "
        "gallery number, tags, and URL. Input: MET object ID integer as a string "
        "(e.g. '437329'). Use METMuseumSearch first to find IDs."
    ),
)
