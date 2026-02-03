from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict

from models.prompts import (
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_ANSWER_SYSTEM_PROMPT,
    DISCUSSION_PLAYER_QUESTION_SYSTEM_PROMPT,
)
from utils.llm import create_chat_llm, get_model, get_provider
from utils.parsing import parse_json_object


_global_discussion_llm = None


def _get_discussion_llm(mock_llm=None):
    """Get or create the discussion LLM instance."""
    global _global_discussion_llm
    if _global_discussion_llm is None:
        _global_discussion_llm = create_chat_llm(role="discussion", temperature=0.2, mock_llm=mock_llm)
    return _global_discussion_llm


def set_discussion_llm(llm):
    """Set a custom LLM for the discussion agent."""
    global _global_discussion_llm
    _global_discussion_llm = llm


def generate_question(
    theme: str,
    last_played_card: int,
    utterances: dict[str, str],
    history: str = "",
    mock_llm=None,
) -> dict:
    """Generates one clarifying question to unblock the game when everyone waits."""
    llm = _get_discussion_llm(mock_llm=mock_llm)

    if llm is None:
        # Mock: simple generic question
        return {"question": "それぞれの発言は、どれくらい強い/大きいイメージですか？"}

    utterances_str = "\n".join([f"{agent}: {word}" for agent, word in utterances.items()])

    prompt = ChatPromptTemplate.from_template(DISCUSSION_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        text = chain.invoke(
            {
                "theme": theme,
                "last_played_card": last_played_card,
                "utterances": utterances_str,
                "history": history,
            }
        )
        result = parse_json_object(text)
        question = str(result.get("question", "")).strip()
        if not question:
            question = "それぞれの発言は、どれくらい強い/大きいイメージですか？"
        return {"question": question}
    except Exception as e:
        print(f"Error generating discussion question: {e}")
        return {"question": "それぞれの発言は、どれくらい強い/大きいイメージですか？"}


def generate_player_question(
    theme: str,
    last_played_card: int,
    utterances: dict[str, str],
    my_word: str,
    history: str = "",
    mock_llm=None,
) -> dict:
    """Generates one question proposal from a player (non-numeric, short)."""
    llm = _get_discussion_llm(mock_llm=mock_llm)

    if llm is None:
        return {"question": "今の発言は、どんなイメージの度合いですか？"}

    utterances_str = "\n".join([f"{agent}: {word}" for agent, word in utterances.items()])

    prompt = ChatPromptTemplate.from_template(DISCUSSION_PLAYER_QUESTION_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        text = chain.invoke(
            {
                "theme": theme,
                "last_played_card": last_played_card,
                "utterances": utterances_str,
                "my_word": my_word,
                "history": history,
            }
        )
        result = parse_json_object(text)
        question = str(result.get("question", "")).strip()
        if not question:
            question = "今の発言は、どんなイメージの度合いですか？"
        return {"question": question}
    except Exception as e:
        print(f"Error generating player discussion question: {e}")
        return {"question": "今の発言は、どんなイメージの度合いですか？"}


def generate_answer(
    theme: str,
    question: str,
    my_word: str,
    history: str = "",
    mock_llm=None,
) -> dict:
    """Generates a short, non-numeric answer to the moderator's question."""
    llm = _get_discussion_llm(mock_llm=mock_llm)

    if llm is None:
        return {
            "answer": "私の発言は、直感的にイメージできる範囲の強さ/大きさを意図しています。"
        }

    prompt = ChatPromptTemplate.from_template(DISCUSSION_ANSWER_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        text = chain.invoke(
            {
                "theme": theme,
                "question": question,
                "my_word": my_word,
                "history": history,
            }
        )
        result = parse_json_object(text)
        answer = str(result.get("answer", "")).strip()
        if not answer:
            answer = "私の発言は、直感的にイメージできる範囲の強さ/大きさを意図しています。"
        return {"answer": answer}
    except Exception as e:
        print(f"Error generating discussion answer: {e}")
        return {
            "answer": "私の発言は、直感的にイメージできる範囲の強さ/大きさを意図しています。"
        }
