"""LLM provider factory.

Returns a chat model for the configured LLM_PROVIDER / LLM_MODEL so the
generator is swappable for the cross-model comparison (DeepSeek, OpenAI,
Anthropic/Claude, Qwen, local Ollama). API keys load from .env.

TODO: implement get_llm()
"""
