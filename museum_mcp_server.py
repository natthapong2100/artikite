
import sys
import os

# Allow importing agents/museum_tools when run from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from mcp.server.fastmcp import FastMCP

import config

_HEADERS = {"User-Agent": "Artikite/1.0 (art history research tool)"}

# ── MCP server instance ───────────────────────────────────────────────────────

mcp = FastMCP(
    "Museum Research Tools",
    instructions=(
        "Use these tools to search the Metropolitan Museum of Art collection. "
        "search_met_artworks finds artworks by query, get_met_artwork returns full "
        "details for a specific object. Always include title, artist, and date in answers."
    ),
)


# ── MET Museum Tools ──────────────────────────────────────────────────────────

@mcp.tool()
def search_met_artworks(query: str, max_results: int = 5) -> str:
    """
    Search the Metropolitan Museum of Art collection.

    Queries the MET's free public API and returns matching artworks with
    title, artist, date, medium, department, and a direct URL.

    Args:
        query:       Artist name, artwork title, medium, period, or any art term.
        max_results: Number of results to return (1–10, default 5).
    """
    max_results = max(1, min(max_results, 10))
    try:
        with httpx.Client(timeout=15, headers=_HEADERS) as client:
            resp = client.get(
                f"{config.MET_API_BASE}/search",
                params={"q": query, "hasImages": True},
            )
            resp.raise_for_status()
            data = resp.json()

        object_ids = (data.get("objectIDs") or [])[:max_results]
        total = data.get("total", 0)

        if not object_ids:
            return f"No artworks found in MET collection for: '{query}'"

        lines = [f"MET Museum — {total} total results for '{query}' (showing {len(object_ids)}):\n"]
        with httpx.Client(timeout=15, headers=_HEADERS) as client:
            for oid in object_ids:
                r = client.get(f"{config.MET_API_BASE}/objects/{oid}")
                if r.status_code == 200:
                    obj = r.json()
                    lines.append(
                        f"• [{oid}] {obj.get('title', 'Untitled')} — "
                        f"{obj.get('artistDisplayName', 'Unknown')} "
                        f"({obj.get('objectDate', '')})\n"
                        f"  Medium: {obj.get('medium', '')} | "
                        f"Dept: {obj.get('department', '')}\n"
                        f"  URL: {obj.get('objectURL', '')}"
                    )
        return "\n".join(lines)
    except Exception as e:
        return f"MET API error: {e}"


@mcp.tool()
def get_met_artwork(object_id: int) -> str:
    """
    Get full details for a specific MET Museum artwork by its object ID.

    Returns title, artist, biography, date, medium, dimensions, department,
    classification, credit line, gallery number, tags, and a direct URL.

    Args:
        object_id: The MET object ID integer (e.g. 437329). Use search_met_artworks
                   first to find IDs.
    """
    try:
        with httpx.Client(timeout=15, headers=_HEADERS) as client:
            resp = client.get(f"{config.MET_API_BASE}/objects/{object_id}")
            resp.raise_for_status()
            obj = resp.json()

        fields = [
            ("Title",          obj.get("title")),
            ("Artist",         obj.get("artistDisplayName")),
            ("Artist Bio",     obj.get("artistDisplayBio")),
            ("Date",           obj.get("objectDate")),
            ("Medium",         obj.get("medium")),
            ("Dimensions",     obj.get("dimensions")),
            ("Department",     obj.get("department")),
            ("Classification", obj.get("classification")),
            ("Credit Line",    obj.get("creditLine")),
            ("Gallery",        obj.get("GalleryNumber")),
            ("URL",            obj.get("objectURL")),
            ("Tags",           ", ".join(t.get("term", "") for t in (obj.get("tags") or []))),
        ]
        lines = [f"MET Artwork #{object_id}"]
        for label, value in fields:
            if value:
                lines.append(f"  {label}: {value}")
        return "\n".join(lines)
    except Exception as e:
        return f"MET API error for object {object_id}: {e}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
