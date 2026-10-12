"""The agent: a LangGraph loop ``START -> agent -> (tools -> agent)* -> END`` with conversation memory.

A checkpointer stores each conversation (thread) so follow-up answers ("by 7:00") reach the agent together
with the earlier messages. The system context (date, calendar, location, app settings) is passed per turn
through the run config and is NOT stored in the conversation, so it never goes stale or piles up.
"""

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from energy_advisor.agent.prompts import SYSTEM_PROMPT, build_system_prompt
from energy_advisor.agent.tools import build_tools
from energy_advisor.config import Settings
from energy_advisor.llm import make_chat_model
from energy_advisor.services.container import Services, build_services
from energy_advisor.timeutils import Clock, calendar_lines

logger = logging.getLogger(__name__)
CALENDAR_DAYS = 8


def build_system_context(now: datetime, context: str | None = None) -> str:
    """Current local date/time and a calendar of the next days, so "tomorrow" or "Wednesday" map to
    exact dates. Today and tomorrow are marked, so "Monday" on a Monday is not misread as today."""
    text = (
        f"Current date and time: {now:%A, %Y-%m-%d %H:%M} ({now.tzname()}).\n"
        f"Upcoming days: {calendar_lines(now, CALENDAR_DAYS)}."
    )
    return f"{text}\n{context}" if context else text


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    instructions: str,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    llm_with_tools = llm.bind_tools(list(tools))

    def agent_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        """Call the LLM with the instructions, this turn's system context and the conversation so far."""
        context = (config.get("configurable") or {}).get("system_context")
        system = [SystemMessage(content=instructions)]
        if context:
            system.append(SystemMessage(content=context))
        return {"messages": [llm_with_tools.invoke([*system, *state["messages"]])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    # handle_tool_errors=True: a failing tool becomes a ToolMessage the LLM can react to
    builder.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(name="energy_advisor", checkpointer=checkpointer)


class EnergyAdvisor:
    """Answers energy-optimization questions using tools (weather, prices, history, RAG, savings)."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        now: Clock,
        instructions: str = SYSTEM_PROMPT,
        default_context: str | None = None,
        recursion_limit: int = 25,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self.instructions = instructions
        self.tools = list(tools)
        self.default_context = default_context
        self._now = now
        self._recursion_limit = recursion_limit
        # in-memory by default: conversations live as long as the process (CLI chat, Streamlit server)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(llm, self.tools, instructions, self.checkpointer)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        services: Services | None = None,
        instructions: str | None = None,
        model: str | None = None,
        optimizer: bool = True,
    ) -> "EnergyAdvisor":
        services = services or build_services(settings)
        return cls(
            llm=make_chat_model(settings, model),
            tools=build_tools(services, optimizer=optimizer),
            instructions=instructions or build_system_prompt(optimizer),
            default_context=(
                f"Household location: {settings.location.name} (time zone {settings.location.timezone}). "
                "All times are local."
            ),
            now=services.clock,
            recursion_limit=settings.llm.recursion_limit,
        )

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex

    def history(self, thread_id: str) -> list[Any]:
        """All stored messages of a conversation (empty for an unknown thread)."""
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(state.values.get("messages", [])) if state and state.values else []

    def invoke(
        self, question: str, context: str | None = None, thread_id: str | None = None
    ) -> dict[str, Any]:
        """Ask a question. Pass the same ``thread_id`` to continue a conversation; without one, a new
        conversation starts.

        Returns the graph state: ``messages`` holds the whole conversation, ``turn_messages`` only this
        turn (the question, tool calls/results and the answer), ``thread_id`` the conversation id.
        The final answer is ``result["messages"][-1].content``.

        Never raises: a failure of the graph returns a normal-shaped result with an apology
        message and an ``error`` key.
        """
        thread_id = thread_id or self.new_thread_id()
        extra = "\n".join(c for c in (self.default_context, context) if c) or None
        config: RunnableConfig = {
            "recursion_limit": self._recursion_limit,
            "configurable": {
                "thread_id": thread_id,
                "system_context": build_system_context(self._now(), extra),
            },
        }
        before = len(self.history(thread_id))
        try:
            result = dict(self.graph.invoke({"messages": [("user", question)]}, config=config))
        except Exception as exc:
            logger.exception("Agent run failed")
            apology = AIMessage(
                content=(
                    "Sorry, I could not complete the analysis because of a technical problem "
                    f"({type(exc).__name__}: {exc}). Please try again."
                )
            )
            return {
                "messages": [apology],
                "turn_messages": [apology],
                "error": str(exc),
                "thread_id": thread_id,
            }
        result["turn_messages"] = result["messages"][before:]
        result["thread_id"] = thread_id
        return result
