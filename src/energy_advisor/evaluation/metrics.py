"""Deterministic tool-usage metrics: did the agent call the tools the question needs?"""

from collections.abc import Sequence

from pydantic import BaseModel


class ToolUsageEval(BaseModel):
    tools_used: list[str]
    expected_tools: list[str]
    appropriateness: float  # share of the used tools that were expected
    completeness: float  # share of the expected tools that were used
    missing_tools: list[str]
    extra_tools: list[str]
    tools_with_errors: list[str]
    feedback: str


def evaluate_tool_usage(
    tools_used: Sequence[str], expected_tools: Sequence[str], tools_with_errors: Sequence[str] = ()
) -> ToolUsageEval:
    used_set, expected_set = set(tools_used), set(expected_tools)
    correct = used_set & expected_set
    missing = sorted(expected_set - used_set)
    extra = sorted(used_set - expected_set)

    feedback = []
    if missing:
        feedback.append(f"Missing expected tools: {missing}.")
    if extra:
        feedback.append(f"Additional tools used (not required, may still be helpful): {extra}.")
    if tools_with_errors:
        feedback.append(f"Tools that returned errors: {sorted(set(tools_with_errors))}.")

    return ToolUsageEval(
        tools_used=list(tools_used),
        expected_tools=list(expected_tools),
        appropriateness=round(len(correct) / len(used_set), 2) if used_set else 0.0,
        completeness=round(len(correct) / len(expected_set), 2) if expected_set else 1.0,
        missing_tools=missing,
        extra_tools=extra,
        tools_with_errors=sorted(set(tools_with_errors)),
        feedback=" ".join(feedback) or "The agent used exactly the expected tools.",
    )


def mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0
