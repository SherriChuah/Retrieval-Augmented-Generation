import argparse
import os
import shutil
import streamlit as st

from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from src.config import (DATA_PATH, CHROMA_PATH)
from .embedding_functions import get_embedding_function


def clear_database() -> None:
    """Clears the chroma database
    """
    print("\n🗑️ Clearing database...")

    if os.path.exists(CHROMA_PATH):
        print("here")
        shutil.rmtree(CHROMA_PATH)


# TODO: maybe expand functionality to other file types?
def load_documents(selected_documents: Optional[str] = os.listdir(DATA_PATH)) -> List[Document]:
    """Loading PDF only documents

    Returns:
        List[Document]: _description_
    """
    
    print("\n👩🏻‍💻 Loading documents...")

    print(selected_documents)
    print(os.listdir(DATA_PATH))

    num_documents = len([f for f in os.listdir(DATA_PATH) 
                         if f.lower().endswith('.pdf') and f in selected_documents])

    missing_content_ocr = set()
    missing_content_muloader = set()
    all_docs = []

    try:
        for item in os.listdir(DATA_PATH):
            print(item)
            if item in selected_documents:
                file_path = os.path.join(DATA_PATH, item)
                if file_path.endswith('.pdf'):
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)

                    for doc in docs:
                        if len(doc.page_content.strip()) < 10:
                            missing_content_muloader.add(doc.metadata["source"])
            
        if missing_content_muloader:
            print(f"\n🔴 Unloadable pdf sources using PyMuPDFLoader and ratio: \
                \nmissing: {missing_content_muloader} \
                \nratio: {round(len(missing_content_muloader)/num_documents, 2)}")
            raise ValueError("\n🟣 Forced exception to try OCR based loader...")
        
    except Exception as e:
        print("\nException: ", e)
        
        for file_path in missing_content_muloader:
            if file_path.endswith('.pdf') and file_path.split("/")[-1] in selected_documents:
                ocr_loader = UnstructuredPDFLoader(file_path)
                docs = ocr_loader.load()
                all_docs.extend(docs)

                for ocr_doc in docs:
                    if len(ocr_doc.page_content.strip()) < 10:
                        missing_content_ocr.add(ocr_doc.metadata["source"])
        
        if missing_content_ocr:
            print(f"\n🔴 Unloadable pdf sources using UnstructuredPDFLoader and ratio: \
                \nmissing: {missing_content_ocr} \
                \nratio: {round(len(missing_content_ocr)/num_documents), 2}")
        else:
            print("\n🟢 All documents loaded...")

    return all_docs


def split_documents_to_chunks(documents: List[Document]) -> List[Document]:
    """Split documents to chunks

    Args:
        documents (List[Document]): 

    Returns:
        List[Document]:
    """
    print("\n䷖ Splitting documents to chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_id(chunks: List[Document]) -> List[Document]:
    """Calculating and assigning chunk ids

    Args:
        chunks (List[Document]): 

    Returns:
        List[Document]:
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get('page')
        current_page_id = f"{source}:{page}"

        # if page id same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata['id'] = chunk_id
    
    return chunks


def add_to_chroma(chunks: List[Document]):
    """Build vector database

    Args:
        chunks (List[Document]): _description_
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # calculate chunk id
    chunks_with_ids = calculate_chunk_id(chunks)

    # add and update the documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"\n▶️ Number of existing documents in DB: {len(existing_ids)} documents")

    # only add documents that dont exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"\n🔵 Adding new documents: {len(new_chunks)} chunks")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        print("Writable?", os.access(CHROMA_PATH, os.W_OK))
        db.add_documents(new_chunks, ids=new_chunks_ids)
    else:
        print("\n🔴 No new documents added.")


def main():
    """Entry point for RAG
    """

    # Clear database if needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        clear_database()


    # Create or update datastore
    documents = load_documents()
    chunks = split_documents_to_chunks(documents)
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()