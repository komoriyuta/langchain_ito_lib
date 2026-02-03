from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.prompts import ESTIMATOR_SYSTEM_PROMPT
from typing import Dict
from utils.llm import create_chat_llm, get_provider, get_model
from utils.parsing import parse_json_object


_global_estimator_llm = None


def _get_estimator_llm(mock_llm=None):
    """Get or create the estimator LLM instance."""
    global _global_estimator_llm
    if _global_estimator_llm is None:
        _global_estimator_llm = create_chat_llm(role="estimator", temperature=0.0, mock_llm=mock_llm)
    return _global_estimator_llm


def set_estimator_llm(llm):
    """Set a custom LLM for the estimator agent."""
    global _global_estimator_llm
    _global_estimator_llm = llm


def decide_action(
    theme: str,
    last_played_card: int,
    utterances: Dict[str, str],
    my_number: int,
    my_word: str,
    history: str = "",
    mock_llm=None,
) -> dict:
    """Decides whether to PLAY or WAIT."""
    llm = _get_estimator_llm(mock_llm=mock_llm)

    if llm is None:
        # Mock logic: Simple heuristic for testing
        # If my number is very small (e.g. < 10) or smaller than some threshold relative to others, PLAY.
        # For simplicity in mock: PLAY if number < 20, else WAIT.
        action = "PLAY" if my_number < 20 else "WAIT"
        return {
            "thought": f"Mock thought: Number is {my_number}, so {action}.",
            "action": action
        }

    # Format utterances for prompt
    utterances_str = "\n".join([f"{agent}: {word}" for agent, word in utterances.items()])

    prompt = ChatPromptTemplate.from_template(ESTIMATOR_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        text = chain.invoke({
            "theme": theme,
            "last_played_card": last_played_card,
            "utterances": utterances_str,
            "my_number": my_number,
            "my_word": my_word,
            "history": history,
        })
        result = parse_json_object(text)
        action = str(result.get("action", "WAIT")).strip().upper()
        if action not in {"PLAY", "WAIT"}:
            action = "WAIT"
        result["action"] = action
        return result
    except Exception as e:
        print(f"Error deciding action: {e}")
        return {"thought": str(e), "action": "WAIT"}  # Default to WAIT on error
