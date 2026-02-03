from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "gemini"]

_global_llm_instances: dict[str, Any] = {}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_provider() -> Provider:
    """Get the configured LLM provider."""
    provider = (os.getenv("ITO_PROVIDER") or "openai").strip().lower()
    if provider in {"gemini"}:
        return "gemini"
    return "openai"


def get_model(role: Literal["speaker", "estimator", "discussion"]) -> str:
    """Get the model name for a given role."""
    # Role-specific override wins
    role_key = f"ITO_{role.upper()}_MODEL"
    if os.getenv(role_key):
        return os.getenv(role_key) or ""

    # Generic override
    if os.getenv("ITO_MODEL"):
        return os.getenv("ITO_MODEL") or ""

    # Provider defaults
    if get_provider() == "gemini":
        return "gemini-2.5-flash-lite"
    return "gpt-4o-mini"


def create_chat_llm(
    *,
    role: Literal["speaker", "estimator", "discussion"],
    temperature: float,
    mock_llm: Any = None,
) -> Any | None:
    """Return a LangChain chat model instance or None (mock mode)."""

    if _is_truthy(os.getenv("ITO_FORCE_MOCK")):
        return None

    if mock_llm is not None:
        return mock_llm

    # Check for singleton instance
    cache_key = f"{get_provider()}:{get_model(role)}"
    if cache_key in _global_llm_instances:
        return _global_llm_instances[cache_key]

    provider = get_provider()
    model = get_model(role)

    # --- API MODE ---
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            return None

        base_url = os.getenv("OPENAI_API_BASE")
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
        )
        _global_llm_instances[cache_key] = llm
        return llm

    # provider == "gemini"
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:  # pragma: no cover
        raise RuntimeError(
            "Gemini provider selected but 'langchain-google-genai' is not installed."
        )

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
    _global_llm_instances[cache_key] = llm
    return llm


def clear_llm_cache() -> None:
    """Clear the LLM instance cache."""
    _global_llm_instances.clear()
