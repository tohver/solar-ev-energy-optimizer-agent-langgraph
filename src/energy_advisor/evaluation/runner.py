"""Run the evaluation cases through the agent and score them."""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import BaseModel

from energy_advisor.agent.graph import EnergyAdvisor
from energy_advisor.agent.trace import final_answer, tool_calls, tool_outputs, tools_with_errors
from energy_advisor.evaluation.cases import EvalCase
from energy_advisor.evaluation.judge import LLMJudge, ResponseEval
from energy_advisor.evaluation.metrics import ToolUsageEval, evaluate_tool_usage

logger = logging.getLogger(__name__)


class CaseResult(BaseModel):
    """Everything logged for one case: the agent's decisions (tool calls), its answer and the scores."""

    test_id: str
    question: str
    answer: str
    tool_calls: list[dict[str, object]]
    expected_tools: list[str]
    expected_response: str
    timestamp: str
    error: str | None = None
    response_eval: ResponseEval
    tool_eval: ToolUsageEval


def run_evaluation(
    advisor: EnergyAdvisor,
    judge: LLMJudge,
    cases: Sequence[EvalCase],
    context: str | None = None,
    now: Callable[[], datetime] = datetime.now,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    results = []
    for case in cases:
        logger.info("Running %s", case.id)
        response = advisor.invoke(case.question, context=context)
        calls = tool_calls(response)
        answer = final_answer(response)
        result = CaseResult(
            test_id=case.id,
            question=case.question,
            answer=answer,
            tool_calls=calls,
            expected_tools=case.expected_tools,
            expected_response=case.expected_response,
            timestamp=now().isoformat(timespec="seconds"),
            error=response.get("error"),
            response_eval=judge.evaluate(
                case.question, answer, case.expected_response, tool_outputs(response), now()
            ),
            tool_eval=evaluate_tool_usage(
                [c["name"] for c in calls], case.expected_tools, tools_with_errors(response)
            ),
        )
        results.append(result)
        if on_result:
            on_result(result)
    return results
