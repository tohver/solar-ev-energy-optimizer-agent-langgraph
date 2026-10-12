"""The agent: a LangGraph loop ``START -> agent -> (tools -> agent)* -> END``."""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from energy_advisor.agent.prompts import SYSTEM_PROMPT
from energy_advisor.agent.tools import build_tools
from energy_advisor.config import Settings
from energy_advisor.llm import make_chat_model
from energy_advisor.services.container import Services, build_services

logger = logging.getLogger(__name__)
CALENDAR_DAYS = 8


def build_system_context(now: datetime, context: str | None = None) -> str:
    """Current date/time and a calendar of the next days, so "tomorrow" or "Wednesday" map to exact dates."""
    calendar = ", ".join(
        f"{(now + timedelta(days=d)):%A} = {(now + timedelta(days=d)):%Y-%m-%d}" for d in range(CALENDAR_DAYS)
    )
    text = f"Current date and time: {now:%A, %Y-%m-%d %H:%M}.\nUpcoming days: {calendar}."
    return f"{text}\n{context}" if context else text


def build_graph(llm: BaseChatModel, tools: Sequence[BaseTool], instructions: str) -> CompiledStateGraph:
    llm_with_tools = llm.bind_tools(list(tools))

    def agent_node(state: MessagesState) -> dict[str, Any]:
        """Call the LLM with the system instructions + the conversation so far."""
        messages = [SystemMessage(content=instructions), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    # handle_tool_errors=True: a failing tool becomes a ToolMessage the LLM can react to
    builder.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(name="energy_advisor")


class EnergyAdvisor:
    """Answers energy-optimization questions using tools (weather, prices, history, RAG, savings)."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        instructions: str = SYSTEM_PROMPT,
        default_context: str | None = None,
        now: Callable[[], datetime] = datetime.now,
        recursion_limit: int = 25,
    ) -> None:
        self.instructions = instructions
        self.tools = list(tools)
        self.default_context = default_context
        self._now = now
        self._recursion_limit = recursion_limit
        self.graph = build_graph(llm, self.tools, instructions)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        services: Services | None = None,
        instructions: str = SYSTEM_PROMPT,
        model: str | None = None,
    ) -> "EnergyAdvisor":
        services = services or build_services(settings)
        return cls(
            llm=make_chat_model(settings, model),
            tools=build_tools(services),
            instructions=instructions,
            default_context=f"Location: {settings.location.name}",
            now=services.now,
            recursion_limit=settings.llm.recursion_limit,
        )

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def invoke(self, question: str, context: str | None = None) -> dict[str, Any]:
        """Ask a question. The final answer is ``result["messages"][-1].content``.

        Never raises: a failure of the graph returns a normal-shaped result with an apology
        message and an ``error`` key.
        """
        system_context = build_system_context(self._now(), context or self.default_context)
        messages = [("system", system_context), ("user", question)]
        try:
            return self.graph.invoke(
                {"messages": messages}, config={"recursion_limit": self._recursion_limit}
            )
        except Exception as exc:
            logger.exception("Agent run failed")
            apology = (
                "Sorry, I could not complete the analysis because of a technical problem "
                f"({type(exc).__name__}: {exc}). Please try again."
            )
            return {"messages": [AIMessage(content=apology)], "error": str(exc)}
