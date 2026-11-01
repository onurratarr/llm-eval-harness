"""LLM-as-judge.

* ``MockJudge`` (default): deterministic heuristic judge so the harness runs offline and CI is reproducible.
  It is a stand-in, not a claim about how a real judge model would score.
* ``OpenAICompatibleJudge``: sends the judge prompt to any OpenAI-compatible endpoint
  (``JUDGE_BACKEND=openai`` + ``OPENAI_API_KEY``; optional ``OPENAI_BASE_URL``, ``JUDGE_MODEL``).

Both return ``{"score": 1-5, "rationale": str}`` and a pairwise verdict ``"A" | "B" | "tie"``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from . import metrics as M
from .rubric import rubric_score

JUDGE_PROMPT = """You are grading an AI assistant's answer.
Question: {prompt}
Reference answer: {reference}
Grading notes: required facts = {required}; must not contain = {forbidden}
Candidate answer: {response}

Score the candidate from 1 (wrong/unhelpful) to 5 (correct, complete, concise). Respond with JSON only:
{{"score": <1-5>, "rationale": "<one sentence>"}}"""

PAIRWISE_PROMPT = """Question: {prompt}
Reference answer: {reference}
Answer A: {a}
Answer B: {b}
Which answer is better (correctness first, then completeness, then concision)? Respond with JSON only:
{{"winner": "A" | "B" | "tie", "rationale": "<one sentence>"}}"""


class MockJudge:
    name = "mock"

    def score(self, item: dict, response: str) -> dict[str, Any]:
        rub, _ = rubric_score(response, item)
        ref = item.get("reference")
        sim = M.token_f1(response, ref) if ref else rub
        raw = 0.6 * rub + 0.4 * sim
        s = 1 + round(4 * raw)
        return {"score": int(s), "rationale": f"mock: rubric={rub:.2f}, ref_f1={sim:.2f}"}

    def pairwise(self, item: dict, a: str, b: str) -> dict[str, Any]:
        sa, sb = self.score(item, a)["score"], self.score(item, b)["score"]
        winner = "tie" if sa == sb else ("A" if sa > sb else "B")
        return {"winner": winner, "rationale": f"mock: A={sa}, B={sb}"}


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"judge returned no JSON: {text[:80]!r}")
    return json.loads(m.group(0))


class OpenAICompatibleJudge:
    name = "openai"

    def __init__(self, post: Callable[[dict], dict] | None = None):
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        self._post = post

    def _ask(self, content: str) -> dict:
        payload = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": content}]}
        if self._post:
            data = self._post(payload)
        else:
            import httpx

            r = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=60,
                           headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
            r.raise_for_status()
            data = r.json()
        return _parse_json(data["choices"][0]["message"]["content"])

    def score(self, item: dict, response: str) -> dict[str, Any]:
        out = self._ask(JUDGE_PROMPT.format(prompt=item["prompt"], reference=item.get("reference", ""),
                                            required=item.get("required", []), forbidden=item.get("forbidden", []),
                                            response=response))
        s = int(out["score"])
        if not 1 <= s <= 5:
            raise ValueError(f"judge score out of range: {s}")
        return {"score": s, "rationale": str(out.get("rationale", ""))}

    def pairwise(self, item: dict, a: str, b: str) -> dict[str, Any]:
        out = self._ask(PAIRWISE_PROMPT.format(prompt=item["prompt"], reference=item.get("reference", ""), a=a, b=b))
        if out.get("winner") not in ("A", "B", "tie"):
            raise ValueError(f"bad pairwise verdict: {out}")
        return {"winner": out["winner"], "rationale": str(out.get("rationale", ""))}


def get_judge(name: str | None = None):
    name = (name or os.environ.get("JUDGE_BACKEND", "mock")).lower()
    return {"mock": MockJudge, "openai": OpenAICompatibleJudge}[name]()
