from langchain_ollama import OllamaEmbeddings


def get_embedding_function() -> OllamaEmbeddings:
    """Embedding funnction used

    Returns:
        OllamaEmbeddings: _description_
    """

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return embeddings