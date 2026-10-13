"""Helpers to inspect an agent result: final answer, tool calls and tool outputs."""

from typing import Any


def final_answer(result: Any) -> str:
    if isinstance(result, dict) and result.get("messages"):
        return str(result["messages"][-1].content)
    return str(result)


def tool_calls(result: Any) -> list[dict[str, Any]]:
    """Tool calls (name + args) made by the agent, in order."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    return [
        {"name": tc["name"], "args": tc["args"]}
        for msg in messages
        for tc in (getattr(msg, "tool_calls", None) or [])
    ]


def tool_outputs(result: Any, per_tool: int = 3000, total: int = 9000) -> str:
    """Compact text of what the tools returned (evidence for the LLM judge)."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    chunks = [
        f"[{m.name}] {str(m.content)[:per_tool]}" for m in messages if getattr(m, "type", None) == "tool"
    ]
    return "\n".join(chunks)[:total] or "(no tools were called)"


def tools_with_errors(result: Any) -> list[str]:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    return sorted(
        {
            m.name
            for m in messages
            if getattr(m, "type", None) == "tool" and "error" in str(m.content).lower()[:200]
        }
    )
