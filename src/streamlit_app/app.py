import streamlit as st
import os

from chat_utils import ChatAgent
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from src.config import MODEL, DATA_PATH, CHROMA_PATH, PROMPT_TEMPLATE
from src.populate_database.embedding_functions import get_embedding_function
from src.populate_database.populate_database import (
    clear_database, load_documents, split_documents_to_chunks, add_to_chroma)


def list_documents():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    return [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]


def reset_and_populate_database(selected_files: list):
    clear_database()

    documents = load_documents(selected_files)
    chunks = split_documents_to_chunks(documents)
    add_to_chroma(chunks)


def save_uploaded_file(uploaded_file):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    with open(os.path.join(DATA_PATH, uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())


def main():
    st.set_page_config(
        page_title="Board Game RAG",
        page_icon='💬',
        layout='wide'
    )

    if "rebuilding_db" not in st.session_state:
        st.session_state.rebuilding_db = False

    col1, col2 = st.columns([1,3])

    with col1:
        st.image('src/streamlit_app/assets/logo.jpg')

        st.markdown('### Manage Documents')

        # Upload pdf
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_file:
            filename = uploaded_file.name
            if filename.endswith(".pdf"):
                save_uploaded_file(uploaded_file)
                st.success(f"Uploaded `{filename}`")

    
        # Drop down multiselect files
        available_pdfs = list_documents()
        selected_pdfs = st.multiselect(
            "Select PDFs to use",
            available_pdfs, default=available_pdfs)
        
        # Apply button
        if st.button("Apply and Rebuild Database"):
            if selected_pdfs:
                st.session_state.rebuilding_db = True

                with st.spinner("Rebuilding vector database..."):
                    reset_and_populate_database(selected_pdfs)

                st.success("Database rebuilt successfully.")
                st.session_state.rebuilding_db = False
            else:
                st.warning("Please select at least one PDF")
        

    with col2:
        st.title("Board Game Instruction Agent")
        st.markdown("""
            #### Example questions
            - Winning condition for Battleship
        """)

        if st.session_state.rebuilding_db:
            st.info("Database is rebuilding. Please wait...")
        else:
            model = OllamaLLM(model=MODEL)
            chat_agent = ChatAgent(llm=model)
            chat_agent.start_conversation()

if __name__ == "__main__":
    main()
