"""Evaluation framework: cases, tool-usage metrics, judge, runner and report."""

import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage

from energy_advisor.agent.graph import EnergyAdvisor
from energy_advisor.agent.tools import build_tools
from energy_advisor.evaluation.cases import EvalCase, load_cases
from energy_advisor.evaluation.judge import LLMJudge, ResponseScores
from energy_advisor.evaluation.metrics import evaluate_tool_usage
from energy_advisor.evaluation.report import generate_report, render_markdown, save_report
from energy_advisor.evaluation.runner import run_evaluation
from tests.helpers import BERLIN, NOW, FakeToolModel


# ----------------------------------------------------------------------------- cases
def test_bundled_cases_are_valid(services):
    cases = load_cases()
    tool_names = {t.name for t in build_tools(services)}
    assert len(cases) >= 10
    assert all(set(c.expected_tools) <= tool_names for c in cases)


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- {id: a, question: q, expected_response: r}\n- {id: a, question: q, expected_response: r}\n"
    )
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


# ----------------------------------------------------------------------------- tool metrics
def test_tool_usage_metrics():
    e = evaluate_tool_usage(
        ["get_weather_forecast", "search_energy_tips", "search_energy_tips"],
        ["get_weather_forecast", "get_electricity_prices"],
        tools_with_errors=["search_energy_tips"],
    )
    assert e.appropriateness == 0.5  # 1 of 2 distinct used tools was expected
    assert e.completeness == 0.5  # 1 of 2 expected tools was used
    assert e.missing_tools == ["get_electricity_prices"]
    assert e.extra_tools == ["search_energy_tips"]
    assert "errors" in e.feedback


def test_tool_usage_perfect_and_empty():
    assert evaluate_tool_usage(["a"], ["a"]).feedback == "The agent used exactly the expected tools."
    nothing = evaluate_tool_usage([], [])
    assert (nothing.appropriateness, nothing.completeness) == (0.0, 1.0)


# ----------------------------------------------------------------------------- judge
class FixedScorer:
    def __init__(self, value: float):
        self.value = value
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        v = self.value
        return ResponseScores(accuracy=v, relevance=v, completeness=v, usefulness=1.5, feedback="fine")


def test_judge_clamps_scores_and_includes_date_in_prompt():
    scorer = FixedScorer(0.8)
    e = LLMJudge(scorer).evaluate("q", "a", "expected", "tools", datetime(2026, 10, 12, tzinfo=BERLIN))
    assert (e.accuracy, e.usefulness) == (0.8, 1.0)
    assert e.overall == pytest.approx(0.85)
    assert "Monday, 2026-10-12" in scorer.prompts[0]


def test_judge_failure_scores_zero_without_raising():
    class Broken:
        def invoke(self, _prompt):
            raise TimeoutError("judge timed out")

    e = LLMJudge(Broken()).evaluate("q", "a", "e", "t", NOW)
    assert e.overall == 0.0
    assert "judge timed out" in e.feedback


# ----------------------------------------------------------------------------- runner + report
@pytest.fixture
def results(services):
    responses = []
    for _ in range(2):
        responses += [
            AIMessage(content="", tool_calls=[{"name": "get_electricity_prices", "args": {}, "id": "c1"}]),
            AIMessage(content="Run it at night."),
        ]
    advisor = EnergyAdvisor(FakeToolModel(messages=iter(responses)), build_tools(services), lambda: NOW)
    cases = [
        EvalCase(id="a", question="q1", expected_tools=["get_electricity_prices"], expected_response="r"),
        EvalCase(
            id="b",
            question="q2",
            expected_tools=["get_electricity_prices", "search_energy_tips"],
            expected_response="r",
        ),
    ]
    seen = []
    out = run_evaluation(advisor, LLMJudge(FixedScorer(0.9)), cases, now=lambda: NOW, on_result=seen.append)
    assert [r.test_id for r in seen] == ["a", "b"]
    return out


def test_runner_logs_tool_calls_and_scores(results):
    assert results[0].tool_calls == [{"name": "get_electricity_prices", "args": {}}]
    assert results[0].answer == "Run it at night."
    assert results[0].tool_eval.completeness == 1.0
    assert results[1].tool_eval.missing_tools == ["search_energy_tips"]


def test_report_aggregates_scores(results):
    report = generate_report(results, now=NOW)
    assert report.total_tests == 2
    assert report.metric_averages["tool_completeness"] == 0.75
    assert report.tests_needing_attention == ["b"]
    assert any("search_energy_tips (1x)" in r for r in report.recommendations)


def test_report_is_rendered_and_saved(results, tmp_path):
    report = generate_report(results, now=NOW)
    md = render_markdown(report, results)
    assert "**Overall score:" in md
    assert "| a |" in md
    paths = save_report(report, results, tmp_path)
    assert paths["markdown"].name == "eval-2026-10-02-093000.md"
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["results"][1]["tool_eval"]["missing_tools"] == ["search_energy_tips"]


def test_report_needs_results():
    with pytest.raises(ValueError, match="No evaluation results"):
        generate_report([])
