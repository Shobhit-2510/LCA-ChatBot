"""LLM provider factory.

Returns a LangChain chat model for the configured LLM_PROVIDER / LLM_MODEL
so the generator is swappable for the cross-model comparison in the plan.
Currently wired for Anthropic (Claude). API keys load from .env.

Note: Opus 4.8 rejects temperature/top_p/top_k (400), so none are set —
ChatAnthropic omits them by default. Extended thinking is left off for low
latency; the prompt instructs answer-only output to keep responses clean.

Public API:
    get_llm() -> BaseChatModel
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv

import config

load_dotenv()  # pull ANTHROPIC_API_KEY (etc.) from .env


@lru_cache(maxsize=1)
def get_llm():
    """Return the chat model for the configured provider."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.LLM_MODEL,          # claude-opus-4-8
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=60,
            max_retries=2,
        )

    # Other providers (deepseek / openai / qwen / ollama) wired here when
    # the plan's model comparison begins.
    raise NotImplementedError(f"LLM provider not yet wired: {provider}")
