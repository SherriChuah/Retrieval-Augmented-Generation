import argparse

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from src.populate_database.embedding_functions import get_embedding_function
from .config import (CHROMA_PATH, PROMPT_TEMPLATE, MODEL)


def query_rag(query_text: str) -> str:
    """Query RAG with text

    Args:
        query_text (str): text to be queries

    Returns:
        str: response in text
    """

    print("🤷🏻‍♀️ Query text: ", query_text)

    # Prepare the db
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )
    print("👾 The prompt")
    print(prompt)

    # Get model and invoke
    model = OllamaLLM(model=MODEL)
    response_text = model.invoke(prompt)

    # Get result and print out
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"""
Response: 

{response_text}

------------------------------------------------
Sources: 

{sources}

"""
    
    print(formatted_response)
    return formatted_response


def main():
    # Create cli
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text...")
    args = parser.parse_args()
    
    # Get the query_text
    query_text = args.query_text
    
    # Query the RAG
    query_rag(query_text)


if __name__ == "__main__":
    main()
