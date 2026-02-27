"""
rag/embedder.py
────────────────
Handles text → vector embedding using Ollama.

Why a separate file?
    The embedding model is independent from the LLM model.
    By isolating it here, you can swap embedding models (e.g., switch from
    nomic-embed-text to mxbai-embed-large) without touching any other file.

How it works:
    Ollama exposes a /api/embeddings endpoint.
    We call it with a text string → get back a float list (the embedding vector).
    ChromaDB stores these vectors and uses cosine similarity to find similar chunks.
"""

import requests
import config


def get_embedding(text: str, model: str = config.OLLAMA_EMBED_MODEL) -> list[float]:
    """
    Call Ollama's embedding endpoint for a single text string.

    Args:
        text:  The text to embed.
        model: Ollama model to use for embedding (default from config).

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError if Ollama is unreachable or returns an error.
    """
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["embedding"]

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except KeyError:
        raise RuntimeError(
            f"Ollama response missing 'embedding' key. "
            f"Make sure you have pulled the model: `ollama pull {model}`"
        )


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts. Ollama doesn't have a native batch endpoint,
    so we call get_embedding() in a loop.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors, same order as input.
    """
    embeddings = []
    for i, text in enumerate(texts):
        print(f"  Embedding {i+1}/{len(texts)}: {text[:50]}...")
        embeddings.append(get_embedding(text))
    return embeddings


def get_embedding_dimension(model: str = config.OLLAMA_EMBED_MODEL) -> int:
    """
    Get the dimension of the embedding vector for the configured model.
    Useful for validating ChromaDB collection settings.
    """
    test_embedding = get_embedding("test", model=model)
    return len(test_embedding)


# ── Simple test when run directly ──
if __name__ == "__main__":
    print("Testing Ollama embeddings...")
    try:
        vec = get_embedding("The Mona Lisa was painted by Leonardo da Vinci.")
        print(f"✅ Embedding successful! Dimension: {len(vec)}")
        print(f"   First 5 values: {vec[:5]}")
    except RuntimeError as e:
        print(f"❌ Error: {e}")
