"""Regression gate: fails if any metric on the synthetic set drops below the stored baseline.

Refresh the baseline intentionally with `python -m llm_eval.cli baseline` and commit the diff.
"""
import copy
import json
from pathlib import Path

from llm_eval.cli import main
from llm_eval.io import read_jsonl
from llm_eval.judge import MockJudge
from llm_eval.runner import check_baseline, run, to_markdown

ROOT = Path(__file__).resolve().parent.parent
ITEMS = read_jsonl(ROOT / "examples/eval_set.jsonl")
LABELS = read_jsonl(ROOT / "examples/pairwise_labels.jsonl")
BASELINE = json.loads((ROOT / "baselines/baseline.json").read_text())


def test_scores_not_below_baseline():
    assert check_baseline(run(ITEMS, LABELS, MockJudge()), BASELINE) == []


def test_degraded_system_is_caught():
    worse = copy.deepcopy(ITEMS)
    for it in worse:
        it["responses"]["sys_a"] = "I am not sure."
    failures = check_baseline(run(worse, LABELS, MockJudge()), BASELINE)
    assert any(f.startswith("sys_a.rubric") for f in failures)
    assert any(f.startswith("sys_a.judge") for f in failures)


def test_cli_check_exit_codes(tmp_path):
    assert main(["check", "--data", str(ROOT / "examples/eval_set.jsonl"), "--labels",
                 str(ROOT / "examples/pairwise_labels.jsonl"), "--baseline", str(ROOT / "baselines/baseline.json")]) == 0
    strict = tmp_path / "b.json"
    strict.write_text(json.dumps({"systems": {"sys_b": {"rubric": 0.99}}}))
    assert main(["check", "--data", str(ROOT / "examples/eval_set.jsonl"), "--baseline", str(strict)]) == 1


def test_committed_results_table_is_current():
    md = to_markdown(run(ITEMS, LABELS, MockJudge()), "examples/eval_set.jsonl")
    assert (ROOT / "results/RESULTS.md").read_text() == md, "re-run `python -m llm_eval.cli run`"
