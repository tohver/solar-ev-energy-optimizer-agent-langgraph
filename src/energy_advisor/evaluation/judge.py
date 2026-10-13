"""LLM-as-judge: a second model scores each answer against the expected response and the tool outputs."""

import logging
from datetime import datetime
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from energy_advisor.timeutils import calendar_lines

logger = logging.getLogger(__name__)

RESPONSE_METRICS = ("accuracy", "relevance", "completeness", "usefulness")

JUDGE_PROMPT = """You are a strict evaluator of an AI energy advisor.
Now: {today}. Calendar: {calendar}.
Dates such as "tomorrow" or weekday names must be consistent with this calendar. Check chronology carefully:
a schedule is only wrong if it ENDS AFTER the user's deadline or starts in the past.
Use the TOOL OUTPUTS as ground truth: penalize accuracy for numbers, dates, hours or cited sources that are not
supported by them (invented sources or wrong dates are serious errors, score accuracy <= 0.5).
Rate the ACTUAL RESPONSE for the QUESTION, using the EXPECTED RESPONSE description as the reference.
Score each metric from 0.0 (very poor) to 1.0 (excellent):
- accuracy: are facts, numbers and calculations plausible, internally consistent and not invented?
- relevance: does it answer exactly what was asked?
- completeness: does it contain all elements described in the expected response (times, costs, savings, citations...)?
- usefulness: is it specific, actionable and quantified (exact hours, temperatures, savings)?
Then give short, concrete feedback.

QUESTION:
{question}

EXPECTED RESPONSE:
{expected}

TOOL OUTPUTS (ground truth, may be truncated):
{tools}

ACTUAL RESPONSE:
{actual}
"""


class ResponseScores(BaseModel):
    """Structured output of the judge (scores from 0.0 to 1.0)."""

    accuracy: float = Field(
        description="Facts, numbers and calculations are correct and consistent with the tool data (0-1)"
    )
    relevance: float = Field(description="The response addresses the user's question directly (0-1)")
    completeness: float = Field(
        description="The response covers everything the expected response requires (0-1)"
    )
    usefulness: float = Field(description="The advice is specific, actionable and quantified (0-1)")
    feedback: str = Field(description="2-4 sentences: strengths, weaknesses and what is missing")


class ResponseEval(BaseModel):
    accuracy: float
    relevance: float
    completeness: float
    usefulness: float
    overall: float
    feedback: str


class Scorer(Protocol):
    """Anything with ``invoke(prompt)``: a structured-output chat model or a test fake."""

    def invoke(self, input: Any, /) -> Any: ...


class LLMJudge:
    def __init__(self, scorer: Scorer) -> None:
        self._scorer = scorer

    @classmethod
    def from_llm(cls, llm: BaseChatModel) -> "LLMJudge":
        return cls(llm.with_structured_output(ResponseScores, method="function_calling"))

    def evaluate(
        self, question: str, answer: str, expected: str, tool_outputs: str, today: datetime
    ) -> ResponseEval:
        """Never raises: a failed judgement scores 0 and says why, so one error doesn't stop the run."""
        try:
            scores = self._scorer.invoke(
                JUDGE_PROMPT.format(
                    question=question,
                    expected=expected,
                    actual=answer,
                    tools=tool_outputs or "(not provided)",
                    today=f"{today:%A, %Y-%m-%d %H:%M}",
                    calendar=calendar_lines(today),
                )
            )
            metrics = {m: round(min(1.0, max(0.0, float(getattr(scores, m)))), 2) for m in RESPONSE_METRICS}
            feedback = scores.feedback
        except Exception as exc:
            logger.warning("Judge failed: %s", exc)
            metrics = dict.fromkeys(RESPONSE_METRICS, 0.0)
            feedback = f"Evaluation failed: {exc}"
        return ResponseEval(
            **metrics, overall=round(sum(metrics.values()) / len(metrics), 2), feedback=feedback
        )
