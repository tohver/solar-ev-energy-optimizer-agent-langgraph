"""Factories for chat models and embeddings, configured from ``Settings``."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from energy_advisor.config import Settings


def make_chat_model(settings: Settings, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.require_api_key(),
        base_url=settings.openai_base_url,
    )


def make_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        api_key=settings.require_api_key(),
        base_url=settings.openai_base_url,
    )
