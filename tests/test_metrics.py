import pytest

from llm_eval import metrics as M
from llm_eval.rubric import rubric_score


def test_reference_metrics():
    assert M.exact_match("80 km/h.", "80 km/h") == 1.0
    assert M.token_f1("the cat sat", "the cat sat down") == pytest.approx(2 * 1 * 0.75 / 1.75)
    assert M.rouge_l("a b c d", "a c d") == pytest.approx(2 * 0.75 * 1 / 1.75)
    assert M.token_f1("", "x") == 0.0


def test_decimals_survive_normalization():
    assert M.normalize("It is 2.5 hours.") == "it is 2.5 hours"


def test_required_uses_word_boundaries():
    assert M.required_coverage("d/dx = 3x^2 + 2x", ["3x^2 + 2"]) == 0.0
    assert M.required_coverage("The derivative is 3x^2 + 2.", ["3x^2 + 2"]) == 1.0


def test_rule_checks():
    assert M.valid_json('{"name": "Maria", "city": "Porto"}', ["name", "city"]) == 1.0
    assert M.valid_json("Name: Maria", ["name"]) == 0.0
    assert M.refusal_correct("I can't share that.", True) == 1.0
    assert M.refusal_correct("Sure, here it is.", True) == 0.0
    assert M.within_length("one two three", 2) == 0.0
    assert M.no_forbidden("Yes you can.", ["yes you can"]) == 0.0


def test_rubric_weights_only_applicable_checks():
    item = {"required": ["30 days", "receipt"], "max_words": 5}
    score, checks = rubric_score("Returns within 30 days with a receipt, refunds in 5 business days.", item)
    assert checks == {"required": 1.0, "length": 0.0}
    assert score == pytest.approx(0.4 / 0.5)
