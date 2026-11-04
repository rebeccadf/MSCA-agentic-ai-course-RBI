import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode


@pytest.mark.parametrize("output, expected", [
    ("1", 1),
    ("  0\n", 0),
    ("1 - matches the gold answer", 1),
    ("correct", 0),  # ambiguous output counts as incorrect
])
def test_judge_parses_output(eval_script, monkeypatch, output, expected):
    async def run(agent, messages):
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr(eval_script.Runner, "run", run)
    assert asyncio.run(eval_script.judge_answer("Q", "gold", "answer")) == expected


def test_evidence_coverage(retrieval_script):
    evidence = "one two three four five six seven"  # 3 five-word sequences
    full = [TextNode(text="intro one two three four five six seven outro")]
    partial = [TextNode(text="one two three four five")]
    assert retrieval_script.evidence_coverage(evidence, full) == 1.0
    assert retrieval_script.evidence_coverage(evidence, partial) == pytest.approx(1 / 3)
    assert retrieval_script.evidence_coverage(evidence, []) == 0.0


def test_evidence_coverage_short_evidence(retrieval_script):
    assert retrieval_script.evidence_coverage("too short", [TextNode(text="too short")]) == 0.0
