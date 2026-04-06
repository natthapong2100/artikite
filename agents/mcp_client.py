"""
Synchronous MCP client — spawns museum_mcp_server.py as a subprocess and
communicates with it via the MCP stdio transport protocol.

This makes MCP the actual transport layer between the pipeline and the museum
server, instead of the pipeline calling the MET REST API directly.

Usage:
    from agents import mcp_client
    mcp_client.connect()                                  # once at startup
    result = mcp_client.call_tool("search_met_artworks", {"query": "Van Gogh"})
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Server location ───────────────────────────────────────────────────────────

_SERVER_PATH = str(Path(__file__).parent.parent / "museum_mcp_server.py")

# ── Module-level state ────────────────────────────────────────────────────────

_loop:     asyncio.AbstractEventLoop | None = None
_session:  ClientSession | None = None
_stdio_cm  = None   # held alive so the subprocess is not killed by GC
_connected = False
_lock      = threading.Lock()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the background event loop, creating it if necessary."""
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True).start()
        return _loop


async def _connect_async() -> None:
    """Open a stdio MCP session to the museum server subprocess."""
    global _session, _stdio_cm
    params = StdioServerParameters(
        command=sys.executable,   # reuse same Python interpreter
        args=[_SERVER_PATH],
    )
    cm = stdio_client(params)
    read, write = await cm.__aenter__()
    _stdio_cm = cm                    # keep reference → subprocess stays alive
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    _session = session


# ── Public API ────────────────────────────────────────────────────────────────

def connect() -> None:
    """
    Spawn museum_mcp_server.py as a subprocess and open a persistent MCP session.
    Safe to call multiple times — only connects once.
    """
    global _connected
    if _connected:
        return
    loop = _ensure_loop()
    asyncio.run_coroutine_threadsafe(_connect_async(), loop).result(timeout=30)
    _connected = True
    print("  [MCP] Connected to Museum MCP Server")


def call_tool(tool_name: str, arguments: dict) -> str:
    """
    Synchronously call an MCP tool on the museum server.
    Auto-connects on first call if connect() was not called explicitly.

    Args:
        tool_name:  MCP tool name (e.g. "search_met_artworks")
        arguments:  Dict of arguments for the tool

    Returns:
        Tool result as a string.
    """
    if not _connected:
        connect()
    try:
        future = asyncio.run_coroutine_threadsafe(
            _session.call_tool(tool_name, arguments),
            _ensure_loop(),
        )
        result = future.result(timeout=30)
        texts = [c.text for c in (result.content or []) if hasattr(c, "text")]
        return "\n".join(texts) if texts else f"[MCP] No result from {tool_name}"
    except Exception as e:
        return f"[MCP] Error calling {tool_name}: {e}"
