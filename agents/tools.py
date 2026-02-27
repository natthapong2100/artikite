"""
agents/tools.py
────────────────
LangChain Tool definitions that wrap the RAG system.

Uses modern LCEL (LangChain Expression Language) style:
    OLD:  LLMChain(llm=llm, prompt=prompt).invoke(...)
    NEW:  (prompt | llm | StrOutputParser()).invoke(...)

The pipe `|` chains: prompt → llm → output parser, same idea but no deprecated classes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from rag.vector_store import retrieve, format_retrieved_context, ingest_documents

# ── Shared state: load collection once ───────────────────────────────────────
_collection = None

def get_collection():
    """Lazy-load the ChromaDB collection (ingest on first call if needed)."""
    global _collection
    if _collection is None:
        _collection = ingest_documents()
    return _collection


# ── Shared LLM + Parser ───────────────────────────────────────────────────────
llm = OllamaLLM(
    model=config.OLLAMA_LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.LLM_TEMPERATURE,
)

parser = StrOutputParser()


# ── Tool Functions ────────────────────────────────────────────────────────────

def art_history_search(query: str) -> str:
    """Core RAG pattern: Retrieve → Augment → Generate."""
    collection = get_collection()
    chunks = retrieve(query, collection=collection, top_k=config.RAG_TOP_K)
    context = format_retrieved_context(chunks)

    prompt = PromptTemplate.from_template(
        """You are an expert art historian. Use the following retrieved knowledge
to answer the question accurately and engagingly.

RETRIEVED KNOWLEDGE:
{context}

QUESTION: {question}

Provide a clear, informative answer based on the retrieved knowledge.
If the knowledge doesn't fully cover the question, say so.

ANSWER:"""
    )

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": query})

    sources = list(set(c["title"] for c in chunks))
    return f"{answer}\n\n📚 Sources: {', '.join(sources)}"


def artist_focused_search(query: str) -> str:
    """Semantic search filtered to artist profiles only."""
    collection = get_collection()
    chunks = retrieve(query, collection=collection,
                      top_k=config.RAG_TOP_K, category_filter="artist")
    context = format_retrieved_context(chunks)

    prompt = PromptTemplate.from_template(
        """You are an art historian specializing in artist biographies and styles.
Using only the artist profiles below, answer the question.

ARTIST PROFILES:
{context}

QUESTION: {question}

Focus on the artist's life, technique, and notable works.

ANSWER:"""
    )

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": query})
    sources = list(set(c["title"] for c in chunks))
    return f"{answer}\n\n🎨 Artists referenced: {', '.join(sources)}"


def movement_focused_search(query: str) -> str:
    """Semantic search filtered to art movements only."""
    collection = get_collection()
    chunks = retrieve(query, collection=collection,
                      top_k=config.RAG_TOP_K, category_filter="movement")
    context = format_retrieved_context(chunks)

    prompt = PromptTemplate.from_template(
        """You are an art historian specializing in art movements and periods.
Using the movement descriptions below, answer the question.

ART MOVEMENT KNOWLEDGE:
{context}

QUESTION: {question}

Explain the movement's key characteristics, historical context, and significance.

ANSWER:"""
    )

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": query})
    sources = list(set(c["title"] for c in chunks))
    return f"{answer}\n\n🏛️ Movements referenced: {', '.join(sources)}"


def compare_art_subjects(subjects: str) -> str:
    """Compare two artists or movements. Input: 'Subject A vs Subject B'"""
    collection = get_collection()

    parts = subjects.split(" vs ")
    if len(parts) != 2:
        return "Please format your comparison as: 'Subject A vs Subject B'"

    subject_a, subject_b = parts[0].strip(), parts[1].strip()

    chunks_a = retrieve(subject_a, collection=collection, top_k=2)
    chunks_b = retrieve(subject_b, collection=collection, top_k=2)
    context_a = format_retrieved_context(chunks_a)
    context_b = format_retrieved_context(chunks_b)

    prompt = PromptTemplate.from_template(
        """You are an expert art historian. Compare {subject_a} and {subject_b}.

KNOWLEDGE ABOUT {subject_a}:
{context_a}

KNOWLEDGE ABOUT {subject_b}:
{context_b}

Write a structured comparison covering:
1. Key similarities
2. Key differences
3. Historical relationship or influence
4. Which is more significant and why

COMPARISON:"""
    )

    chain = prompt | llm | parser
    return chain.invoke({
        "subject_a": subject_a, "subject_b": subject_b,
        "context_a": context_a, "context_b": context_b
    })


# ── Register as LangChain Tools ───────────────────────────────────────────────

art_search_tool = Tool(
    name="ArtHistorySearch",
    func=art_history_search,
    description="""Search the art history knowledge base for any question about
    artists, artworks, movements, or periods. Input: a natural language question.
    Example: 'What technique did Caravaggio use?' or 'Tell me about Impressionism'"""
)

artist_search_tool = Tool(
    name="ArtistSearch",
    func=artist_focused_search,
    description="""Search specifically for information about individual artists.
    Use this for biographical details, artistic style, or notable works.
    Input: a question about a specific artist."""
)

movement_search_tool = Tool(
    name="MovementSearch",
    func=movement_focused_search,
    description="""Search specifically for information about art movements and periods.
    Input: a question about an art movement."""
)

compare_tool = Tool(
    name="ArtComparison",
    func=compare_art_subjects,
    description="""Compare two artists or art movements side by side.
    Input MUST be in format: 'Subject A vs Subject B'
    Example: 'Leonardo da Vinci vs Michelangelo' or 'Cubism vs Surrealism'"""
)

ALL_TOOLS = [art_search_tool, artist_search_tool, movement_search_tool, compare_tool]