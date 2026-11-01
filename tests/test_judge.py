import json

import pytest

from llm_eval.judge import MockJudge, OpenAICompatibleJudge

ITEM = {"id": "t", "prompt": "Convert 2.5 hours to minutes.", "reference": "150 minutes", "required": ["150 minutes"]}


def test_mock_judge_is_deterministic_and_ordered():
    j = MockJudge()
    good, bad = j.score(ITEM, "150 minutes"), j.score(ITEM, "120 minutes")
    assert good == j.score(ITEM, "150 minutes")
    assert good["score"] == 5 and bad["score"] < good["score"]
    assert j.pairwise(ITEM, "120 minutes", "150 minutes")["winner"] == "B"


def _reply(obj):
    return lambda payload: {"choices": [{"message": {"content": json.dumps(obj)}}]}


def test_openai_judge_parses_and_validates():
    j = OpenAICompatibleJudge(post=_reply({"score": 4, "rationale": "ok"}))
    assert j.score(ITEM, "150 minutes") == {"score": 4, "rationale": "ok"}
    with pytest.raises(ValueError):
        OpenAICompatibleJudge(post=_reply({"score": 9})).score(ITEM, "x")
    with pytest.raises(ValueError):
        OpenAICompatibleJudge(post=_reply({"winner": "C"})).pairwise(ITEM, "a", "b")
    assert OpenAICompatibleJudge(post=_reply({"winner": "tie"})).pairwise(ITEM, "a", "b")["winner"] == "tie"
