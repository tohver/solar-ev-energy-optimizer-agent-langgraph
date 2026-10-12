"""Constants and fakes shared by the tests."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage

FIXTURES = Path(__file__).parent / "fixtures"
LAST_SEEDED_DAY = date(2026, 10, 1)  # the test database ends on this day
NOW = datetime(2026, 10, 2, 9, 30)  # the clock injected into services and the agent


class FakeToolModel(GenericFakeChatModel):
    """Scripted chat model that supports ``bind_tools`` and records what it was sent."""

    received: list[list[BaseMessage]] = []  # noqa: RUF012 - pydantic field, copied per instance

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)
