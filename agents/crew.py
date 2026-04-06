import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent, Task, Crew, Process, LLM

import config


def _detect_sources(query: str) -> list[str]:
    """Route using the same LLM logic as LangGraph."""
    from agents.langgraph_flow import _route_sources
    return _route_sources(query)


def _build_tool_instructions(sources: list[str]) -> str:
    """Return the tool-use paragraph for the research task based on routing decision."""
    if "met" in sources:
        return (
            "Use your METMuseumSearch tool to retrieve live artworks from the Metropolitan Museum."
        )
    return (
        "Use your ArtHistorySearch, ArtistSearch, and MovementSearch tools to gather "
        "comprehensive information from the art history knowledge base."
    )


from crewai.tools import tool as crewai_tool
from agents.tools import art_history_search, artist_focused_search, movement_focused_search, compare_art_subjects
from agents.museum_tools import met_search

@crewai_tool("METMuseumSearch")
def met_search_tool(query: str) -> str:
    """Search the Metropolitan Museum of Art collection for real artworks, artists, and periods."""
    return met_search(query)

@crewai_tool("ArtHistorySearch")
def art_search_tool(query: str) -> str:
    """Search the art history knowledge base for any question about artists, artworks, movements, or periods."""
    return art_history_search(query)

@crewai_tool("ArtistSearch")
def artist_search_tool(query: str) -> str:
    """Search specifically for information about individual artists, their biography, style, and notable works."""
    return artist_focused_search(query)

@crewai_tool("MovementSearch")
def movement_search_tool(query: str) -> str:
    """Search specifically for information about art movements and periods, their characteristics and context."""
    return movement_focused_search(query)

@crewai_tool("ArtComparison")
def compare_tool(subjects: str) -> str:
    """Compare two artists or art movements. Input MUST be in format: 'Subject A vs Subject B'"""
    return compare_art_subjects(subjects)



# ── Ollama LLM for CrewAI ─────────────────────────────────────────────────────

# crewai_llm = LLM(
#     model=f"ollama/{config.OLLAMA_LLM_MODEL}",
#     base_url=config.OLLAMA_BASE_URL,
#     temperature=config.LLM_TEMPERATURE,
# )

crewai_llm = LLM(
    model=f"ollama/{config.OLLAMA_LLM_MODEL}",
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.LLM_TEMPERATURE,
    api_key="ollama",   # litellm requires a non-empty string even for local models
)


# ── Agent Definitions ─────────────────────────────────────────────────────────

def create_research_specialist() -> Agent:
    """
    Agent 1: The Research Specialist
    Responsible for gathering factual information from the art history knowledge base.
    Has access to all RAG search tools.
    """
    return Agent(
        role="Art History Research Specialist",
        goal="Find accurate, comprehensive factual information about art history topics "
             "using the knowledge base. Focus on specific artists, artworks, dates, and "
             "historical context.",
        backstory="""You are a meticulous art history researcher with a PhD in Art History
        from the Courtauld Institute of Art. You have spent 15 years cataloguing artworks
        in major museums. You are known for your precision — you always cite specific
        artworks, dates, and sources. You never make up facts. When information isn't
        in the knowledge base, you say so clearly.""",
        llm=crewai_llm,
        tools=[art_search_tool, artist_search_tool, movement_search_tool, met_search_tool],
        verbose=config.CREW_VERBOSE,
        allow_delegation=False,  # Researchers don't delegate — they do the work
        max_execution_time=60,
    )


def create_art_critic() -> Agent:
    """
    Agent 2: The Art Critic
    Responsible for cultural analysis and critical interpretation.
    Uses the researcher's output to write insightful analysis.
    """
    return Agent(
        role="Art Critic and Cultural Analyst",
        goal="Transform factual research into  cultural analysis. Intcompellingerpret "
             "artworks in their historical context, identify themes and influences, "
             "and explain why the subject matters to art history.",
        backstory="""You are a celebrated art critic who writes for major publications
        like The Art Newspaper and Artforum. With 20 years of experience, you have
        a gift for making art history accessible and compelling. You combine rigorous
        scholarship with vivid storytelling. You believe art cannot be understood
        without its historical and cultural context. You love finding unexpected
        connections between artists and movements.""",
        llm=crewai_llm,
        tools=[compare_tool],  # Can use comparison tool for analysis
        verbose=config.CREW_VERBOSE,
        allow_delegation=False,
        max_execution_time=60,
    )


def create_museum_curator() -> Agent:
    """
    Agent 3: The Museum Curator
    Responsible for final review, polishing, and ensuring educational value.
    Can delegate back to the critic if the analysis needs improvement.
    """
    return Agent(
        role="Senior Museum Curator and Editor",
        goal="Review and polish the analysis to ensure it is accurate, well-structured, "
             "educationally valuable, and appropriate for a museum audience. Ensure all "
             "claims are supported and the narrative flows logically.",
        backstory="""You are the Chief Curator at a major art museum with 25 years of
        experience creating exhibitions and educational programs. You have exceptionally
        high standards — every piece of content under your name must be both accurate
        and engaging. You are known for your ability to spot gaps in arguments and for
        elevating good writing into excellent writing. You can delegate to the art critic
        if the analysis requires significant improvements.""",
        llm=crewai_llm,
        tools=[art_search_tool],  # Can do fact-checking lookups
        verbose=config.CREW_VERBOSE,
        allow_delegation=False,  # if True, Curator CAN delegate back to the research specialist
        max_execution_time=90,
    )


# ── Task Definitions ──────────────────────────────────────────────────────────

def create_tasks(query: str, research_agent: Agent,
                 critic_agent: Agent, curator_agent: Agent,
                 sources: list[str] | None = None) -> list[Task]:
    """
    Create the three tasks for the crew.
    Note how each task builds on the previous via the `context` parameter.
    """
    if sources is None:
        sources = _detect_sources(query)

    tool_instructions = _build_tool_instructions(sources)
    print(f"     🧭 CrewAI router chose: {sources}")

    research_task = Task(
        description=f"""Research the following art history topic thoroughly: "{query}"

        {tool_instructions}

        Your research MUST include:
        1. Key artists involved (names, dates, nationalities)
        2. Specific artworks mentioned (titles, dates, current locations if known)
        3. The art movement(s) involved and their defining characteristics
        4. Historical context (what was happening politically/socially at the time)
        5. Any notable techniques or innovations

        Be specific. Vague generalities are not acceptable.""",

        expected_output="""A structured research report with:
        - Artist profiles (biography highlights, key works, techniques)
        - Movement/period overview (dates, characteristics, key figures)
        - Historical context
        - Specific artworks with dates
        All facts sourced from the knowledge base.""",

        agent=research_agent,
    )

    analysis_task = Task(
        description=f"""Using the research findings, write a compelling cultural analysis
        of: "{query}"

        You have access to a comparison tool if you need to compare artists or movements.

        Your analysis MUST:
        1. Open with an engaging hook that draws the reader in
        2. Explain WHY this topic matters in art history
        3. Analyze the artistic techniques and their significance
        4. Discuss the cultural and historical forces that shaped the subject
        5. Identify the lasting influence or legacy
        6. Include at least one insightful comparison or contrast

        Write for an intelligent general audience — assume curiosity but not expertise.
        Length: 4–6 substantial paragraphs.""",

        expected_output="""A well-crafted cultural analysis essay with:
        - Engaging introduction
        - Substantive body paragraphs with specific details
        - Discussion of historical significance
        - Clear conclusion with lasting impact
        Written in an accessible yet authoritative tone.""",

        agent=critic_agent,
        context=[research_task],  # This task sees research_task's output
    )

    curation_task = Task(
        description=f"""Review and finalize the cultural analysis about: "{query}"

        You have access to ArtHistorySearch to fact-check specific claims.
        You may delegate to the Art Critic if substantial rewriting is needed.

        Review checklist:
        □ All factual claims are accurate (cross-check if uncertain)
        □ Specific artworks and dates are mentioned
        □ The analysis answers the original question fully
        □ The narrative flows logically from introduction to conclusion
        □ Tone is appropriate (engaging but scholarly)
        □ No unexplained jargon

        Produce the final polished version. Add a brief 'Curator's Note' at the end
        summarizing 2–3 key takeaways for visitors.""",

        expected_output="""The final, publication-ready analysis including:
        - Polished essay with any corrections applied
        - 'Curator's Note' with 2-3 key takeaways
        - Ready for museum publication or academic use""",

        agent=curator_agent,
        context=[research_task, analysis_task],  # Sees both previous outputs
    )

    return [research_task, analysis_task, curation_task]


# ── Public API ────────────────────────────────────────────────────────────────

def run_crew_analysis(query: str) -> str:
    """
    Assemble and run the Art History Crew for a given query.

    Args:
        query: The art history question to analyze.

    Returns:
        The final curated analysis as a string.
    """
    print(f"\n  Assembling Art History Crew for: '{query}'")

    # Detect which sources to use (same keyword logic as LangGraph router)
    sources = _detect_sources(query)

    # Create agents
    research_agent = create_research_specialist()
    critic_agent = create_art_critic()
    curator_agent = create_museum_curator()

    # Create tasks with routing-aware instructions
    tasks = create_tasks(query, research_agent, critic_agent, curator_agent, sources=sources)

    # Assemble crew
    crew = Crew(
        agents=[research_agent, critic_agent, curator_agent],
        tasks=tasks,
        process=Process.sequential,   # Tasks run in order: research → analyze → curate
        verbose=config.CREW_VERBOSE,
    )

    # Kick off the crew
    print("  🚀 Crew is starting work...\n")
    result = crew.kickoff()

    return str(result)
