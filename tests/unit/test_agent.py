"""Agent graph tested with a scripted fake LLM: no network, no cost, deterministic."""

import inspect
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage

from energy_advisor.agent.graph import EnergyAdvisor, build_system_context
from energy_advisor.agent.tools import build_tools
from energy_advisor.agent.trace import final_answer, tool_calls, tool_outputs, tools_with_errors
from tests.helpers import BERLIN, NOW, FakeToolModel

EXPECTED_TOOLS = {
    "get_weather_forecast",
    "get_electricity_prices",
    "query_energy_usage",
    "query_solar_generation",
    "get_recent_energy_summary",
    "find_cheapest_window",
    "estimate_pv_generation",
    "search_energy_tips",
    "calculate_energy_savings",
}


def _advisor(services, *responses, **kwargs):
    llm = FakeToolModel(messages=iter(responses))
    advisor = EnergyAdvisor(llm, build_tools(services), lambda: NOW, **kwargs)
    return advisor, llm


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_toolkit_has_all_nine_tools(services):
    assert {t.name for t in build_tools(services)} == EXPECTED_TOOLS


def test_every_tool_has_a_docstring_for_the_llm(services):
    assert all(len(t.description) > 40 for t in build_tools(services))


def test_invoke_signature_keeps_question_and_context():
    assert {"question", "context"} <= set(inspect.signature(EnergyAdvisor.invoke).parameters)


def test_graph_has_agent_and_tools_nodes(services):
    advisor, _ = _advisor(services)
    assert {"agent", "tools"} <= set(advisor.graph.get_graph().nodes)


def test_system_context_contains_local_date_calendar_and_location():
    text = build_system_context(datetime(2026, 10, 12, 9, 5, tzinfo=BERLIN), "Location: Munich")
    assert "Current date and time: Monday, 2026-10-12 09:05 (CEST)." in text
    assert "Monday = 2026-10-12 (today)" in text
    assert "Tuesday = 2026-10-13 (tomorrow)" in text
    assert "Monday = 2026-10-19" in text  # 8 days of calendar: next Monday is distinguishable
    assert text.endswith("Location: Munich")


def test_agent_calls_tool_then_answers(services):
    advisor, _ = _advisor(
        services,
        _tool_call("get_electricity_prices", {"date": "2026-10-03"}),
        AIMessage(content="Charge between 22:00 and 06:00."),
    )
    result = advisor.invoke("When should I charge my EV tomorrow?")

    assert final_answer(result) == "Charge between 22:00 and 06:00."
    assert tool_calls(result) == [{"name": "get_electricity_prices", "args": {"date": "2026-10-03"}}]
    assert "hourly_rates" in tool_outputs(result)
    assert "error" not in result


def test_system_prompt_and_context_are_sent_to_the_llm(services):
    advisor, llm = _advisor(
        services, AIMessage(content="ok"), instructions="ROLE: test", default_context="Location: X"
    )
    advisor.invoke("hi")
    sent = llm.received[0]
    assert sent[0].content == "ROLE: test"
    assert "2026-10-02" in sent[1].content
    assert "Location: X" in sent[1].content
    assert sent[-1].content == "hi"


def test_tool_error_is_reported_to_the_llm_not_raised(services):
    advisor, _ = _advisor(
        services,
        _tool_call("get_electricity_prices", {"date": "14/10/2026"}),
        AIMessage(content="Sorry, the date was invalid."),
    )
    result = advisor.invoke("Prices?")
    assert tools_with_errors(result) == ["get_electricity_prices"]
    assert final_answer(result) == "Sorry, the date was invalid."


def test_graph_failure_returns_apology_instead_of_raising(services):
    advisor, _ = _advisor(services)  # the fake model has no scripted answer -> it fails
    result = advisor.invoke("Anything")
    assert "error" in result
    assert final_answer(result).startswith("Sorry, I could not complete the analysis")


def test_from_settings_needs_an_api_key(settings, services):
    from energy_advisor.config import MissingApiKeyError

    with pytest.raises(MissingApiKeyError):
        EnergyAdvisor.from_settings(settings.model_copy(update={"openai_api_key": None}), services)


def test_ablation_removes_the_optimizer_tools_and_prompt_sections(services):
    from energy_advisor.agent.prompts import build_system_prompt

    without = {t.name for t in build_tools(services, optimizer=False)}
    assert without == EXPECTED_TOOLS - {"find_cheapest_window", "estimate_pv_generation"}
    assert "find_cheapest_window" not in build_system_prompt(optimizer=False)
    assert "find_cheapest_window" in build_system_prompt(optimizer=True)


# ----------------------------------------------------------------------------- conversation memory
def test_follow_up_answer_reaches_the_agent_with_the_earlier_conversation(services):
    advisor, llm = _advisor(
        services,
        AIMessage(content="By what time do you need the car?"),
        AIMessage(content="Then charge from 01:00 to 04:00."),
    )
    first = advisor.invoke("When should I charge my EV with 30 kWh?")
    second = advisor.invoke("By 7:00 tomorrow.", thread_id=first["thread_id"])

    assert final_answer(second) == "Then charge from 01:00 to 04:00."
    seen = [m.content for m in llm.received[1]]
    assert "When should I charge my EV with 30 kWh?" in seen
    assert "By what time do you need the car?" in seen
    assert [m.content for m in second["turn_messages"]] == [
        "By 7:00 tomorrow.",
        "Then charge from 01:00 to 04:00.",
    ]
    assert len(second["messages"]) == 4


def test_conversations_are_isolated(services):
    advisor, llm = _advisor(services, AIMessage(content="a"), AIMessage(content="b"))
    advisor.invoke("first question")
    advisor.invoke("second question")  # no thread id: a new conversation
    assert "first question" not in [m.content for m in llm.received[1]]


def test_system_context_is_sent_every_turn_but_not_stored(services):
    advisor, llm = _advisor(services, AIMessage(content="a"), AIMessage(content="b"))
    first = advisor.invoke("q1", context="Window: X")
    advisor.invoke("q2", thread_id=first["thread_id"])
    assert "Window: X" in llm.received[0][1].content
    assert "Window: X" not in llm.received[1][1].content  # context of turn 1 does not leak into turn 2
    assert all(m.type != "system" for m in advisor.history(first["thread_id"]))
