import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Make sure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from rag.vector_store import ingest_documents
from agents.langgraph_flow import run_langgraph_research
from agents.crew import run_crew_analysis
from agents import mcp_client


# ── Sample queries (great for interview demos) ────────────────────────────────
SAMPLE_QUERIES = [
    "How did Leonardo da Vinci revolutionize Renaissance painting?",
    "Compare Baroque and Impressionism as artistic movements",
    "What makes Van Gogh's style unique and why was he so influential?",
    "How did Cubism change the way we think about art and perspective?",
    "Tell me about the relationship between Frida Kahlo and Surrealism",
    "Who were the most important artists of the Italian Renaissance?",
]


def print_banner():
    print("\n" + "═" * 65)
    print("  🎨  Art History RAG Agentic System")
    print("  📚  LangChain + LangGraph + CrewAI + ChromaDB + Ollama")
    print("═" * 65)


def print_section(title: str, char: str = "─"):
    print(f"\n{char * 65}")
    print(f"  {title}")
    print(f"{char * 65}")


def save_report(query: str, langgraph_result: dict, crew_result: str) -> str:
    """Save the combined output to a markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query[:40].replace(" ", "_").replace("?", "").replace(",", "")
    filename = f"report_{safe_query}_{timestamp}.md"
    filepath = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)

    content = f"""# Art History Analysis Report

**Query:** {query}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**System:** LangChain + LangGraph + CrewAI + ChromaDB + Ollama (Llama3.1)

---

## Part 1: LangGraph Research Pipeline

> *Orchestrated workflow: Plan → Research → Write → Validate → [Revise]*

**Workflow log:**
{chr(10).join(f"- {entry}" for entry in langgraph_result["log"])}

**Validation status:** {langgraph_result["validation_status"]}  
**Revision iterations:** {langgraph_result["iteration"]}

### Essay

{langgraph_result["final_essay"]}

---

## Part 2: CrewAI Multi-Agent Analysis

> *Three specialized agents: Research Specialist → Art Critic → Museum Curator*

{crew_result}

---

## System Architecture Notes

| Component | Role |
|-----------|------|
| **LangChain** | LLM interface, PromptTemplates, LLMChains, Tool objects |
| **LangGraph** | Stateful graph workflow with conditional routing and loops |
| **CrewAI** | Multi-agent collaboration with role-based specialization |
| **ChromaDB** | Vector database for semantic retrieval |
| **Ollama** | Local Llama3.1 inference for LLM + embeddings |
| **RAG** | Retrieval-Augmented Generation combining search + generation |
"""

    with open(filepath, "w") as f:
        f.write(content)

    return filepath


def run(query: str = None):
    """
    Main pipeline. Pass a query string or leave None to use interactive mode.
    """
    print_banner()

    # ── Step 1: Initialize ChromaDB ──────────────────────────────────────────
    print_section("STEP 1: Initializing ChromaDB Vector Store", "═")
    print("\n  Loading art history knowledge base...")
    collection = ingest_documents()
    print(f"  ✅ Vector store ready with {collection.count()} chunks")

    # ── Step 2: Connect to Museum MCP Server ─────────────────────────────────
    print_section("STEP 2: Connecting to Museum MCP Server", "═")
    mcp_client.connect()
    print("  ✅ MCP session ready")

    # ── Step 3: Get query ────────────────────────────────────────────────────
    if query is None:
        print_section("STEP 3: Choose Your Query", "═")
        print("\n  Sample queries:")
        for i, q in enumerate(SAMPLE_QUERIES, 1):
            print(f"  [{i}] {q}")
        print(f"  [0] Enter your own question")

        choice = input("\n  Enter number (or press Enter for default): ").strip()

        if choice == "0":
            query = input("  Your question: ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(SAMPLE_QUERIES):
            query = SAMPLE_QUERIES[int(choice) - 1]
        else:
            query = SAMPLE_QUERIES[0]
            print(f"  Using default: {query}")

    print(f"\n  📝 Query: {query}")

    # ── Steps 3 & 4: LangGraph + CrewAI in parallel ─────────────────────────
    # print_section("STEPS 3 & 4: LangGraph + CrewAI running in parallel", "═")
    # print("  LangGraph: Planner → Researcher → Writer → Validator → [Reviser]")
    # print("  CrewAI:    Research Specialist → Art Critic → Museum Curator")
    # print("  (both pipelines running at the same time)\n")

    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     f_langgraph = executor.submit(run_langgraph_research, query)
    #     f_crew      = executor.submit(run_crew_analysis, query)
    #     langgraph_result = f_langgraph.result()
    #     crew_result      = f_crew.result()
    
    
    print_section("STEPS 4 & 5: LangGraph → CrewAI running sequentially", "═")
    print("  LangGraph: Planner → Researcher → Writer → Validator → [Reviser]")
    print("  CrewAI:    Research Specialist → Art Critic → Museum Curator\n")

    langgraph_result = run_langgraph_research(query)
    crew_result      = run_crew_analysis(query)

    print_section("LangGraph Final Essay", "─")
    print(langgraph_result["final_essay"])

    print_section("CrewAI Final Analysis", "─")
    print(crew_result)

    # ── Step 6: Save Report ──────────────────────────────────────────────────
    print_section("STEP 6: Saving Report", "═")
    report_path = save_report(query, langgraph_result, crew_result)
    print(f"\n  💾 Report saved to: {report_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("PIPELINE COMPLETE", "═")
    print(f"""
  Query:          {query}
  LangGraph:      {len(langgraph_result['log'])} steps, {langgraph_result['iteration']} revision(s)
  Validation:     {langgraph_result['validation_status']}
  CrewAI:         3 agents (Researcher → Critic → Curator)
  Report saved:   {report_path}

  ─────────────────────────────────────────────────────────────
  Frameworks used:
  ✅ LangChain  — LLM abstraction, tools, prompt chains
  ✅ LangGraph  — Stateful workflow with conditional routing
  ✅ CrewAI     — Role-based multi-agent collaboration
  ✅ ChromaDB   — Vector DB for semantic retrieval (RAG)
  ✅ Ollama     — Local inference (no API keys needed)
  ✅ MCP        — Museum tool transport (MET API via MCP protocol)
""")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # You can pass a query directly here, or set to None for interactive mode
    QUERY = None   # Set to e.g. "Tell me about Van Gogh" to skip the menu

    run(query=QUERY)
