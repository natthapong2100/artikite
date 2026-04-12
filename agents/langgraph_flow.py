import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from agents.tools import art_search_tool, artist_search_tool, movement_search_tool
from agents.museum_tools import met_search_tool, met_get_artwork_tool


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_think(text: str) -> str:
    """Remove qwen3 <think>...</think> reasoning blocks before using LLM output.
    Also handles truncated blocks where </think> was cut off by num_predict."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # complete blocks
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)           # truncated blocks
    return text.strip()


# ── LLM + Parser (shared across all nodes) ───────────────────────────────────
llm = OllamaLLM(
    model=config.OLLAMA_LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.LLM_TEMPERATURE,
    num_predict=config.LLM_MAX_TOKENS,
)
parser = StrOutputParser()


# ── State Definition ──────────────────────────────────────────────────────────

class ArtResearchState(TypedDict):
    query: str
    research_plan: str
    raw_research: str
    draft_essay: str
    final_essay: str
    validation_feedback: str
    validation_status: str
    iteration: int
    log: List[str]


# ── Node Functions ────────────────────────────────────────────────────────────

def planner_node(state: ArtResearchState) -> ArtResearchState:
    """Node 1: Break the query into a structured research plan."""
    print("\n  🗺️  [LangGraph] PLANNER node running...")

    prompt = PromptTemplate.from_template(
        """You are a senior art history research coordinator.
A student asked: "{query}"

Create a focused research plan. List exactly 3 specific questions to investigate
that will comprehensively answer the student's query.

Format strictly as:
RESEARCH PLAN:
Q1: [specific question about an artist or artwork]
Q2: [specific question about historical context or movement]
Q3: [specific question about influence or significance]"""
    )

    chain = prompt | llm | parser
    plan = strip_think(chain.invoke({"query": state["query"]}))

    print(f"     Plan: {plan[:100]}...")
    return {
        **state,
        "research_plan": plan,
        "log": state["log"] + ["PLANNER: Research plan created"]
    }


def _route_sources(query: str) -> list[str]:
    """
    Use the LLM to decide which data source(s) to query.
    The LLM understands institution names, places, and context.
    qwen3 emits <think>…</think> tokens — stripped before parsing.
    """
    import re

    prompt = PromptTemplate.from_template(
        """You are a routing agent for an art history research system.

Question: {query}

Sources:
- chromadb : general art history knowledge — use for questions about movements, periods, styles, biographies, historical context, or influence
- met      : Metropolitan Museum of Art live collection — use ONLY when the question specifically asks what the MET has, owns, or holds

Rules:
- If the question asks about a specific museum's collection or holdings, use ONLY that museum's source
- If the question is about art history in general (even if it mentions a known artist), use chromadb
- If the question combines both (e.g. "Tell me about Van Gogh and what the MET has"), use both


Output ONLY a comma-separated list. No explanation.
Valid tokens: chromadb, met

Answer:"""
    )
    chain = prompt | llm | parser
    raw = chain.invoke({"query": query})

    # Strip qwen3 <think>…</think> reasoning block if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    valid = {"chromadb", "met"}
    chosen = [s.strip().lower() for s in raw.replace('"', '').split(",")]
    sources = [s for s in chosen if s in valid]
    print(f"     Parsed sources: {sources}")
    return sources or ["chromadb", "met"]  # fallback: general knowledge


def researcher_node(state: ArtResearchState) -> ArtResearchState:
    """Node 2: Route to the right data sources (ChromaDB / MET / MOMA) then research."""
    print("\n  🔍  [LangGraph] RESEARCHER node running...")

    query = state["query"]

    # ── Step 1: LLM decides which sources to query ────────────────────────────
    sources = _route_sources(query)
    print(f"     🧭 Router chose: {sources}")

    # ── Step 2: Query only the selected sources ───────────────────────────────
    parts = []

    if "chromadb" in sources:
        print("     → ArtHistorySearch (ChromaDB RAG)...")
        parts.append("=== ART HISTORY KNOWLEDGE BASE ===\n" + art_search_tool.func(query))
        print("     → ArtistSearch (ChromaDB RAG)...")
        parts.append("=== ARTIST PROFILES ===\n" + artist_search_tool.func(query))
        print("     → MovementSearch (ChromaDB RAG)...")
        parts.append("=== MOVEMENTS & PERIODS ===\n" + movement_search_tool.func(query))

    if "met" in sources:
        print("     → METMuseumSearch (live API)...")
        search_results = met_search_tool.func(query)
        parts.append("=== MET MUSEUM COLLECTION (live) ===\n" + search_results)
        ids = re.findall(r'\[(\d+)\]', search_results)
        if ids:
            print(f"     → METGetArtwork (live API, ID {ids[0]})...")
            parts.append("=== MET ARTWORK DETAIL ===\n" + met_get_artwork_tool.func(ids[0]))

    compiled = "\n\n".join(parts)
    print(f"     Research compiled: {len(compiled)} chars from {sources}")
    return {
        **state,
        "raw_research": compiled,
        "log": state["log"] + [f"RESEARCHER: Queried {sources}"]
    }


def writer_node(state: ArtResearchState) -> ArtResearchState:
    """Node 3: Synthesize research into a structured essay."""
    print("\n  ✍️   [LangGraph] WRITER node running...")

    prompt = PromptTemplate.from_template(
        """You are an engaging art history writer for a general educated audience.

Original Question: {query}

Research Plan:
{plan}

Retrieved Research:
{research}

Write a well-structured art history analysis with these sections:
## Introduction
## Key Findings
## Historical Significance
## Conclusion

Write in an engaging, authoritative tone. Cite specific artworks and dates."""
    )

    chain = prompt | llm | parser
    draft = strip_think(chain.invoke({
        "query": state["query"],
        "plan": state["research_plan"],
        "research": state["raw_research"]
    }))

    print(f"     Draft written: {len(draft)} chars")
    return {
        **state,
        "draft_essay": draft,
        "log": state["log"] + ["WRITER: Draft essay created"]
    }


def validator_node(state: ArtResearchState) -> ArtResearchState:
    """Node 4: Review quality and set routing status."""
    print("\n  ✅  [LangGraph] VALIDATOR node running...")

    # ── Heuristic pre-checks (fast, no LLM) ──────────────────────────────────
    draft = state["draft_essay"]
    issues = []
    if len(draft) < 300:
        issues.append("Essay is too short (under 300 characters).")
    # required_sections = ["## Introduction", "## Key Findings", "## Historical Significance", "## Conclusion"]
    # missing = [s for s in required_sections if s not in draft]
    
    draft_lower = draft.lower()
    required_sections = ["## introduction", "## key findings", "## historical significance", "## conclusion"]
    missing = [s for s in required_sections if s not in draft_lower]
    
    if missing:
        issues.append(f"Missing required sections: {', '.join(missing)}.")

    if issues:
        feedback = "Pre-check failed: " + " ".join(issues)
        print(f"     Pre-check FAILED: {feedback}")
        return {
            **state,
            "validation_feedback": feedback,
            "validation_status": "NEEDS_IMPROVEMENT",
            "final_essay": draft,
            "log": state["log"] + [f"VALIDATOR: Pre-check failed — {feedback}"]
        }

    prompt = PromptTemplate.from_template(
        """You are a strict art history editor. Review this draft essay.

Original Question: {query}

Draft Essay:
{draft}

Evaluate on:
1. Does it fully answer the question?
2. Are specific artworks, dates, and artists mentioned?
3. Is the structure clear (intro, findings, significance, conclusion)?
4. Is the tone appropriate for educated readers?

Reply with EXACTLY one of:
VALID - if the essay is complete and high quality
NEEDS_IMPROVEMENT - if it requires more detail or corrections

Then provide 1-2 sentences of specific feedback.

Your review:"""
    )

    chain = prompt | llm | parser
    review = strip_think(chain.invoke({
        "query": state["query"],
        "draft": state["draft_essay"]
    }))

    status = "VALID" if ("VALID" in review.upper() and "NEEDS_IMPROVEMENT" not in review.upper()) else "NEEDS_IMPROVEMENT"

    print(f"     Validation: {status}")
    return {
        **state,
        "validation_feedback": review,
        "validation_status": status,
        "final_essay": state["draft_essay"],
        "log": state["log"] + [f"VALIDATOR: Status = {status}"]
    }


def reviser_node(state: ArtResearchState) -> ArtResearchState:
    """Node 5: Improve draft based on validator feedback."""
    print(f"\n  🔄  [LangGraph] REVISER node (iteration {state['iteration'] + 1})...")

    prompt = PromptTemplate.from_template(
        """You are revising an art history essay based on editorial feedback.

Original Question: {query}

Current Draft:
{draft}

Editorial Feedback:
{feedback}

Rewrite the essay addressing all the feedback. Add more specific detail about
artworks, dates, and historical context where needed."""
    )

    chain = prompt | llm | parser
    revised = strip_think(chain.invoke({
        "draft": state["draft_essay"],
        "feedback": state["validation_feedback"],
        "query": state["query"]
    }))

    return {
        **state,
        "draft_essay": revised,
        "iteration": state["iteration"] + 1,
        "log": state["log"] + [f"REVISER: Revision {state['iteration'] + 1} complete"]
    }


# ── Conditional Edge ──────────────────────────────────────────────────────────

def route_after_validation(state: ArtResearchState) -> str:
    if (state["validation_status"] == "NEEDS_IMPROVEMENT"
            and state["iteration"] < config.MAX_REVISION_LOOPS):
        return "revise"
    return "done"


# ── Build & Compile Graph ─────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ArtResearchState)

    graph.add_node("planner",    planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("validator",  validator_node)
    graph.add_node("reviser",    reviser_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer",     "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"revise": "reviser", "done": END}
    )
    graph.add_edge("reviser", "validator")

    return graph.compile()


art_research_graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_langgraph_research(query: str) -> dict:
    """Run the full LangGraph research workflow."""
    print(f"\n  Starting LangGraph workflow for: '{query}'")

    initial_state = ArtResearchState(
        query=query,
        research_plan="",
        raw_research="",
        draft_essay="",
        final_essay="",
        validation_feedback="",
        validation_status="",
        iteration=0,
        log=[]
    )

    result = art_research_graph.invoke(initial_state)

    print(f"\n  📋 LangGraph Log:")
    for entry in result["log"]:
        print(f"     → {entry}")

    return result