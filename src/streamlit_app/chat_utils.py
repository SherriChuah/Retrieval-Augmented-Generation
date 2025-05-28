import streamlit as st

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from src.config import MODEL, CHROMA_PATH, PROMPT_TEMPLATE
from src.populate_database.embedding_functions import get_embedding_function


class ChatAgent:
    def __init__(self, llm: Runnable):
        """Initialise Chat Agent

        Args:
            prompt (ChatPromptTemplate): _description_
            llm (Runnable): _description_
        """
        self.history = StreamlitChatMessageHistory(key="chat_history")
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.setup_chain()
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )

    def fetch_docs_fn(self, inputs: dict) -> dict:
        """Get top k related chunks to the input question.

        Args:
            inputs (dict): dictionary with 'question' as key

        Returns:
            dict: dictionary with other important info for output
        """
        docs_with_score = self.db.similarity_search_with_score(inputs["question"], k=5)
        docs = [doc for doc, _ in docs_with_score]
        context = "\n\n--\n\n".join(doc.page_content for doc in docs)

        return {
            "context": context,
            "question": inputs["question"],
            "sources": [doc.metadata for doc in docs],
            "docs": docs
        }
    

    def format_response_fn(self, inputs_and_response: dict) -> dict:
        """Format for response to query

        Args:
            inputs_and_response (dict): Contains 'inputs' and 'llm_output' key

        Returns:
            dict: dictionary other important info for output 
        """
        llm_output = inputs_and_response['llm_output']
        original_input = inputs_and_response['inputs']

        return {
            "docs": original_input["docs"],
            "sources": original_input["sources"],
            "output": llm_output,
        }

    
    def setup_chain(self) -> RunnableWithMessageHistory:
        """Set up the chain for Chat Agent

        Returns:
            RunnableWithMessageHistory: Configured chain with message history
        """
        fetch_docs = RunnableLambda(self.fetch_docs_fn) 

        format_response = RunnableLambda(self.format_response_fn) 

        def run_llm_with_context(inputs):
            formatted_input = self.prompt.invoke({
                "context": inputs["context"],
                "question": inputs["question"]
            })
            llm_output = self.llm.invoke(formatted_input)
            return {
                "inputs": inputs,           
                "llm_output": llm_output 
            }
        
        run_llm = RunnableLambda(run_llm_with_context)


        # 2. Full chain: retrieve ➝ format prompt ➝ LLM
        chain = fetch_docs | run_llm | format_response

        return RunnableWithMessageHistory(
            chain,
            lambda _: self.history,
            input_messages_key="question",
            history_messages_key="history",
        )
    

    def display_messages(self):
        """Display message in chat interface. Add default AI if no message
        """
        if len(self.history.messages) == 0:
            self.history.add_ai_message("How can I help you?")

        for msg in self.history.messages:
            st.chat_message(msg.type).write(msg.content)


    def start_conversation(self):
        self.display_messages()

        user_question = st.chat_input(placeholder="Ask me anything...")

        if user_question:
            st.chat_message("human").write(user_question)
            config = {"configurable": {"session_id": "any"}}
            result = self.chain.invoke(
                {"question": user_question},
                config
            )

            st.chat_message("ai").write(result['output'])

            with st.expander("Sources"):
                for i, (src, doc) in enumerate(zip(result["sources"], result["docs"]), 1):
                    st.markdown(f"Doc {i}: {doc} \n\n Sources: \n {src}")
                    