MODEL = 'mistral'

DATA_PATH = "src/data"

CHROMA_PATH = "src/chroma"

PROMPT_TEMPLATE = """

Answer the question based only on the following context:

{context}

____________

Answer the question based on the above context: {question}
"""

EVAL_PROMPT = """

Expected Response: {expected_response}

Actual Response: {actual_response}

____________

(Answer with 'true' or 'false') Does the expected response match the actual response?
"""