from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.prompts import SPEAKER_SYSTEM_PROMPT
from utils.llm import create_chat_llm, get_provider, get_model
from utils.parsing import parse_json_object


_global_speaker_llm = None


def _get_speaker_llm(mock_llm=None):
    """Get or create the speaker LLM instance."""
    global _global_speaker_llm
    if _global_speaker_llm is None:
        _global_speaker_llm = create_chat_llm(role="speaker", temperature=0.7, mock_llm=mock_llm)
    return _global_speaker_llm


def set_speaker_llm(llm):
    """Set a custom LLM for the speaker agent."""
    global _global_speaker_llm
    _global_speaker_llm = llm


def generate_word(theme: str, number: int, history: str = "", mock_llm=None) -> dict:
    """Generates a word based on the theme and number."""
    llm = _get_speaker_llm(mock_llm=mock_llm)

    if llm is None:
        # Mock logic
        return {
            "word": f"Mock Word (Number: {number})",
            "reasoning": "Mock reasoning because API key is missing."
        }

    prompt = ChatPromptTemplate.from_template(SPEAKER_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        text = chain.invoke({
            "theme": theme,
            "number": number,
            "history": history,
        })
        return parse_json_object(text)
    except Exception as e:
        print(f"Error generating word: {e}")
        return {"word": "Error", "reasoning": str(e)}
