import pytest

from llm_eval.agreement import align, cohen_kappa, interpret_kappa, to_preference_pairs


def test_cohen_kappa_known_values():
    assert cohen_kappa(list("AABB"), list("ABBB")) == pytest.approx(0.5)
    assert cohen_kappa(list("ABAB"), list("ABAB")) == 1.0
    # Classic worked example: 50 items, po=0.7, pe=0.5 -> kappa 0.4
    a = ["Y"] * 25 + ["N"] * 25
    b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_validation_and_bands():
    with pytest.raises(ValueError):
        cohen_kappa(["A"], ["A", "B"])
    assert interpret_kappa(0.5) == "moderate" and interpret_kappa(0.9) == "almost perfect"


def test_align_and_invalid_label():
    labels = [{"pair_id": "p1", "annotator": "x", "label": "A"}, {"pair_id": "p1", "annotator": "y", "label": "B"},
              {"pair_id": "p2", "annotator": "x", "label": "tie"}]
    assert align(labels, "x", "y") == (["p1"], ["A"], ["B"])
    with pytest.raises(ValueError):
        align([{"pair_id": "p", "annotator": "x", "label": "C"}], "x", "y")


def test_preference_pairs_majority_and_ties_dropped():
    pairs = [{"pair_id": p, "prompt": "q", "response_a": "a", "response_b": "b"} for p in ("p1", "p2", "p3")]
    labels = [{"pair_id": "p1", "annotator": "x", "label": "B"}, {"pair_id": "p1", "annotator": "y", "label": "B"},
              {"pair_id": "p2", "annotator": "x", "label": "A"}, {"pair_id": "p2", "annotator": "y", "label": "B"},
              {"pair_id": "p3", "annotator": "x", "label": "tie"}]
    out = to_preference_pairs(pairs, labels)
    assert out == [{"pair_id": "p1", "prompt": "q", "chosen": "b", "rejected": "a", "votes": {"B": 2}}]
