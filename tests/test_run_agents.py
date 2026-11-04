import ast
import asyncio
from types import SimpleNamespace

import pytest


def evaluate(script, expression):
    return script._eval_node(ast.parse(expression, mode="eval").body)


@pytest.mark.parametrize("expression, expected", [
    ("1 + 2 * 3", 7),
    ("(1200 + 1350) / 2", 1275),
    ("-5 + +2", -3),
    ("2 ** 10", 1024),
    ("7118 / 5964", 7118 / 5964),
])
def test_calculator_evaluates_arithmetic(agents_script, expression, expected):
    assert evaluate(agents_script, expression) == pytest.approx(expected)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hi')",
    "abs(-1)",
    "x + 1",
    "'a' + 'b'",
    "7 // 2",
])
def test_calculator_rejects_anything_but_arithmetic(agents_script, expression):
    with pytest.raises(ValueError):
        evaluate(agents_script, expression)


def test_lookup_budget(agents_script):
    qctx = agents_script.QuestionContext(doc_name="X", retriever=None, search_top_k=6)
    allowed = [agents_script._take_lookup(qctx) for _ in range(agents_script.MAX_LOOKUPS + 2)]
    assert allowed == [True] * agents_script.MAX_LOOKUPS + [False, False]


def test_results_file_round_trip(agents_script, tmp_path):
    results_file = tmp_path / "results.json"
    assert agents_script.load_results(results_file) == []
    agents_script.append_result(results_file, {"financebench_id": "a"})
    agents_script.append_result(results_file, {"financebench_id": "b"})
    assert agents_script.load_results(results_file) == [{"financebench_id": "a"}, {"financebench_id": "b"}]

# ------------------------------------------------------------------
# answer_question routing, with the agents replaced by canned outputs
# ------------------------------------------------------------------

@pytest.fixture
def fake_agents(agents_script, monkeypatch):
    """Replace run_agent with canned outputs per agent; records which agents ran."""
    s = agents_script
    calls = []
    outputs = {
        "Router": s.Route(kind="calculation"),
        "Extractor": s.Facts(facts=[], missing=[]),
        "Calculator": '{"computed_metrics": {"capex": 1.0}}',
        "Composer": "Capex was $1.0 million.",
        "Analyst": "Analyst answer.",
    }

    async def run_agent(agent, messages, context=None, attempts=3):
        calls.append(agent.name)
        out = outputs[agent.name]
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(final_output=out)

    monkeypatch.setattr(s, "run_agent", run_agent)
    return SimpleNamespace(calls=calls, outputs=outputs)


def answer(agents_script):
    qctx = agents_script.QuestionContext(doc_name="X", retriever=None, search_top_k=6)
    return asyncio.run(agents_script.answer_question("Q?", "context", qctx))


def test_calculation_route_runs_full_pipeline(agents_script, fake_agents):
    assert answer(agents_script) == ("calculation", "Capex was $1.0 million.")
    assert fake_agents.calls == ["Router", "Extractor", "Calculator", "Composer"]


def test_text_route_goes_to_analyst(agents_script, fake_agents):
    fake_agents.outputs["Router"] = agents_script.Route(kind="text")
    assert answer(agents_script) == ("text", "Analyst answer.")
    assert fake_agents.calls == ["Router", "Analyst"]


def test_insufficient_data_falls_back_to_analyst(agents_script, fake_agents):
    fake_agents.outputs["Calculator"] = '{"error": "insufficient data"}'
    assert answer(agents_script) == ("calculation->text", "Analyst answer.")
    assert fake_agents.calls == ["Router", "Extractor", "Calculator", "Analyst"]


def test_extractor_out_of_turns_falls_back_to_analyst(agents_script, fake_agents):
    fake_agents.outputs["Extractor"] = agents_script.MaxTurnsExceeded("too many turns")
    assert answer(agents_script) == ("calculation->text", "Analyst answer.")
    assert fake_agents.calls == ["Router", "Extractor", "Analyst"]
