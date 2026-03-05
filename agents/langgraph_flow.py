import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from agents.tools import art_search_tool, artist_search_tool, movement_search_tool


# ── LLM + Parser (shared across all nodes) ───────────────────────────────────
llm = OllamaLLM(
    model=config.OLLAMA_LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.LLM_TEMPERATURE,
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
    plan = chain.invoke({"query": state["query"]})

    print(f"     Plan: {plan[:100]}...")
    return {
        **state,
        "research_plan": plan,
        "log": state["log"] + ["PLANNER: Research plan created"]
    }


def researcher_node(state: ArtResearchState) -> ArtResearchState:
    """Node 2: Execute research using RAG tools (ChromaDB)."""
    print("\n  🔍  [LangGraph] RESEARCHER node running...")

    query = state["query"]

    print("     → Calling ArtHistorySearch (RAG)...")
    general_info = art_search_tool.func(query)

    print("     → Calling ArtistSearch (RAG, artist filter)...")
    artist_info = artist_search_tool.func(query)

    print("     → Calling MovementSearch (RAG, movement filter)...")
    movement_info = movement_search_tool.func(query)

    compiled = f"""
=== GENERAL ART HISTORY CONTEXT ===
{general_info}

=== ARTIST INFORMATION ===
{artist_info}

=== MOVEMENT & PERIOD CONTEXT ===
{movement_info}
""".strip()

    print(f"     Research compiled: {len(compiled)} chars")
    return {
        **state,
        "raw_research": compiled,
        "log": state["log"] + ["RESEARCHER: Retrieved content from ChromaDB via 3 RAG tools"]
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
    draft = chain.invoke({
        "query": state["query"],
        "plan": state["research_plan"],
        "research": state["raw_research"]
    })

    print(f"     Draft written: {len(draft)} chars")
    return {
        **state,
        "draft_essay": draft,
        "log": state["log"] + ["WRITER: Draft essay created"]
    }


def validator_node(state: ArtResearchState) -> ArtResearchState:
    """Node 4: Review quality and set routing status."""
    print("\n  ✅  [LangGraph] VALIDATOR node running...")

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
    review = chain.invoke({
        "query": state["query"],
        "draft": state["draft_essay"]
    })

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
    revised = chain.invoke({
        "draft": state["draft_essay"],
        "feedback": state["validation_feedback"],
        "query": state["query"]
    })

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