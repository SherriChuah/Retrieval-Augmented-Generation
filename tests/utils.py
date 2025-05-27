from src.query import query_rag
from langchain_ollama import OllamaLLM

from src.config import (EVAL_PROMPT, MODEL)


def query_and_validate(question: str, expected_response: str) -> bool:
    response_text = query_rag(question)

    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text
    )

    model = OllamaLLM(model=MODEL)
    evaluation_result_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_result_str.strip().lower()

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
    