"""Rubric scoring: weighted combination of rule-based checks, configured per item.

An item may specify any of: ``required`` (facts that must appear), ``forbidden`` (phrases that must not), ``max_words``,
``json_keys`` (output must be a JSON object with these keys) and ``should_refuse``. Only the checks that apply to an item
are scored; weights are renormalised over them.
"""
from __future__ import annotations

from . import metrics as M

WEIGHTS = {"required": 0.4, "forbidden": 0.2, "length": 0.1, "json": 0.2, "refusal": 0.1}


def rubric_checks(pred: str, item: dict) -> dict[str, float]:
    checks: dict[str, float] = {}
    if item.get("required"):
        checks["required"] = M.required_coverage(pred, item["required"])
    if item.get("forbidden"):
        checks["forbidden"] = M.no_forbidden(pred, item["forbidden"])
    if item.get("max_words"):
        checks["length"] = M.within_length(pred, item["max_words"])
    if item.get("json_keys"):
        checks["json"] = M.valid_json(pred, item["json_keys"])
    if "should_refuse" in item:
        checks["refusal"] = M.refusal_correct(pred, item["should_refuse"])
    return checks


def rubric_score(pred: str, item: dict) -> tuple[float, dict[str, float]]:
    checks = rubric_checks(pred, item)
    if not checks:
        return 1.0, checks
    total_w = sum(WEIGHTS[k] for k in checks)
    return round(sum(WEIGHTS[k] * v for k, v in checks.items()) / total_w, 4), checks
