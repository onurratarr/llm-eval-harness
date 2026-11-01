"""Inter-annotator agreement and pairwise-preference utilities."""
from __future__ import annotations

from collections import Counter

PAIRWISE_LABELS = ("A", "B", "tie")


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters over the same items (nominal labels)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and the same length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def percent_agreement(a: list, b: list) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) bands."""
    for limit, label in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= limit:
            return label
    return "almost perfect"


def align(labels: list[dict], rater_a: str, rater_b: str) -> tuple[list, list, list]:
    """Pair two raters' labels on the items both labelled. Rows: {pair_id, annotator, label}."""
    by = {}
    for r in labels:
        if r["label"] not in PAIRWISE_LABELS:
            raise ValueError(f"invalid label {r['label']!r} for {r['pair_id']}")
        by[(r["pair_id"], r["annotator"])] = r["label"]
    ids = sorted({p for p, ann in by if ann == rater_a} & {p for p, ann in by if ann == rater_b})
    return ids, [by[(i, rater_a)] for i in ids], [by[(i, rater_b)] for i in ids]


def to_preference_pairs(pairs: list[dict], labels: list[dict]) -> list[dict]:
    """Convert pairwise labels to RLHF/DPO-style {prompt, chosen, rejected} rows.

    Uses the majority label per pair; ties and pairs without a majority are dropped.
    """
    votes: dict[str, Counter] = {}
    for r in labels:
        votes.setdefault(r["pair_id"], Counter())[r["label"]] += 1
    out = []
    for p in pairs:
        c = votes.get(p["pair_id"])
        if not c:
            continue
        (top, n), *rest = c.most_common()
        if top == "tie" or (rest and rest[0][1] == n):
            continue
        chosen, rejected = (p["response_a"], p["response_b"]) if top == "A" else (p["response_b"], p["response_a"])
        out.append({"pair_id": p["pair_id"], "prompt": p["prompt"], "chosen": chosen, "rejected": rejected,
                    "votes": dict(c)})
    return out
