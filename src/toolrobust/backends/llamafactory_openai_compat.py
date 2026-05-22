"""Placeholder OpenAI-compatible backend for future local LLaMA-Factory services."""

from __future__ import annotations

from toolrobust.backends.openai_chat import OpenAIChatBackend


class LlamaFactoryOpenAICompatBackend(OpenAIChatBackend):
    """Thin adapter for OpenAI-compatible local endpoints."""

    @property
    def name(self) -> str:
        return "llamafactory_openai_compat"
