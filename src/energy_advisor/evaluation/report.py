"""Aggregate case results into a report (scores, strengths, weaknesses, recommendations) and save it."""

import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from energy_advisor.evaluation.judge import RESPONSE_METRICS
from energy_advisor.evaluation.metrics import mean
from energy_advisor.evaluation.runner import CaseResult

TOOL_METRICS = ("tool_appropriateness", "tool_completeness")
ALL_METRICS = (*RESPONSE_METRICS, *TOOL_METRICS)
STRONG, WEAK = 0.8, 0.7


class TestRow(BaseModel):
    __test__ = False  # not a pytest test class

    test_id: str
    accuracy: float
    relevance: float
    completeness: float
    usefulness: float
    response_overall: float
    tool_appropriateness: float
    tool_completeness: float


class EvaluationReport(BaseModel):
    generated_at: str
    total_tests: int
    overall_score: float
    response_score: float
    tool_score: float
    metric_averages: dict[str, float]
    per_test: list[TestRow]
    strengths: list[str]
    weaknesses: list[str]
    best_tests: list[str]
    tests_needing_attention: list[str]
    recommendations: list[str]


def _row(r: CaseResult) -> TestRow:
    e = r.response_eval
    return TestRow(
        test_id=r.test_id,
        accuracy=e.accuracy,
        relevance=e.relevance,
        completeness=e.completeness,
        usefulness=e.usefulness,
        response_overall=e.overall,
        tool_appropriateness=r.tool_eval.appropriateness,
        tool_completeness=r.tool_eval.completeness,
    )


def generate_report(results: Sequence[CaseResult], now: datetime | None = None) -> EvaluationReport:
    if not results:
        raise ValueError("No evaluation results")
    rows = [_row(r) for r in results]
    averages = {m: mean([getattr(row, m) for row in rows]) for m in ALL_METRICS}
    response_score = mean([row.response_overall for row in rows])
    tool_score = round((averages["tool_appropriateness"] + averages["tool_completeness"]) / 2, 2)

    recommendations = []
    if averages["tool_completeness"] < 0.9:
        missing = Counter(t for r in results for t in r.tool_eval.missing_tools)
        recommendations.append(
            "Strengthen the system prompt so the agent calls all required tools; most often skipped: "
            + ", ".join(f"{k} ({v}x)" for k, v in missing.most_common())
            + "."
        )
    if averages["accuracy"] < STRONG:
        recommendations.append(
            "Improve accuracy: require numbers to come from tool outputs and use calculate_energy_savings with real rates."
        )
    if averages["completeness"] < STRONG:
        recommendations.append(
            "Improve completeness: require the structure Recommendation -> Why -> Savings -> Tips with source citations."
        )
    if averages["usefulness"] < STRONG:
        recommendations.append(
            "Improve usefulness: demand exact hours, setpoints and savings estimates in every answer."
        )
    if not recommendations:
        recommendations.append(
            "Performance is strong; next steps: broaden the knowledge base and add edge-case tests."
        )

    return EvaluationReport(
        generated_at=(now or datetime.now()).isoformat(timespec="seconds"),
        total_tests=len(rows),
        overall_score=round((response_score + tool_score) / 2, 2),
        response_score=response_score,
        tool_score=tool_score,
        metric_averages=averages,
        per_test=rows,
        strengths=[f"{m}: {v:.2f}" for m, v in averages.items() if v >= STRONG],
        weaknesses=[f"{m}: {v:.2f}" for m, v in averages.items() if v < WEAK],
        best_tests=[row.test_id for row in sorted(rows, key=lambda x: x.response_overall, reverse=True)[:3]],
        tests_needing_attention=[
            row.test_id for row in rows if row.response_overall < WEAK or row.tool_completeness < 1.0
        ],
        recommendations=recommendations,
    )


def render_markdown(report: EvaluationReport, results: Sequence[CaseResult] = ()) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Generated: {report.generated_at} · Tests: {report.total_tests}",
        "",
        f"**Overall score: {report.overall_score:.2f} / 1.00** "
        f"(response quality {report.response_score:.2f}, tool usage {report.tool_score:.2f})",
        "",
        "## Metric averages",
        "",
        "| Metric | Score |",
        "|---|---|",
        *[f"| {m} | {v:.2f} |" for m, v in report.metric_averages.items()],
        "",
        f"- **Strengths:** {'; '.join(report.strengths) or 'none above 0.80'}",
        f"- **Weaknesses:** {'; '.join(report.weaknesses) or 'none below 0.70'}",
        f"- **Best tests:** {', '.join(report.best_tests)}",
        f"- **Need attention:** {', '.join(report.tests_needing_attention) or 'none'}",
        "",
        "## Recommendations",
        "",
        *[f"{i}. {rec}" for i, rec in enumerate(report.recommendations, 1)],
        "",
        "## Per-test results",
        "",
        "| Test | " + " | ".join(ALL_METRICS) + " |",
        "|---|" + "---|" * len(ALL_METRICS),
        *[
            f"| {row.test_id} | " + " | ".join(f"{getattr(row, m):.2f}" for m in ALL_METRICS) + " |"
            for row in report.per_test
        ],
    ]
    if results:
        lines += ["", "## Feedback per test", ""]
        for r in results:
            lines += [
                f"### {r.test_id}",
                "",
                f"*{r.question}*",
                "",
                f"- Tools used: {', '.join(c['name'] for c in r.tool_calls) or 'none'}",
                f"- Response: {r.response_eval.feedback}",
                f"- Tools: {r.tool_eval.feedback}",
                "",
            ]
    return "\n".join(lines) + "\n"


def save_report(report: EvaluationReport, results: Sequence[CaseResult], out_dir: Path) -> dict[str, Path]:
    """Write ``eval-<timestamp>.md`` and ``.json``; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "eval-" + report.generated_at.replace(":", "").replace("T", "-")
    md_path, json_path = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    md_path.write_text(render_markdown(report, results), encoding="utf-8")
    payload = {
        "report": report.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}
