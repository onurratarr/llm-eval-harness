"""llm-eval command line.

    python -m llm_eval.cli run       --data examples/eval_set.jsonl --labels examples/pairwise_labels.jsonl
    python -m llm_eval.cli check     --baseline baselines/baseline.json     # exit 1 on regression
    python -m llm_eval.cli baseline  --out baselines/baseline.json          # refresh the stored baseline
    python -m llm_eval.cli kappa     --labels examples/pairwise_labels.jsonl
    python -m llm_eval.cli export-preferences --out results/preferences.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

from .agreement import align, cohen_kappa, interpret_kappa
from .io import read_jsonl, write_jsonl
from .runner import check_baseline, make_baseline, run, save, to_markdown

DATA = "examples/eval_set.jsonl"
LABELS = "examples/pairwise_labels.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check", "baseline", "export-preferences"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=DATA)
        p.add_argument("--labels", default=LABELS)
        p.add_argument("--judge", default=None, help="mock (default) or openai")
        p.add_argument("--systems", default="sys_a,sys_b")
        if name == "run":
            p.add_argument("--out", default="results/results.json")
            p.add_argument("--md", default="results/RESULTS.md")
            p.add_argument("--rows", default="results/rows.jsonl")
        if name in ("check", "baseline"):
            p.add_argument("--baseline", default="baselines/baseline.json")
            p.add_argument("--tolerance", type=float, default=0.02)
        if name == "baseline":
            p.add_argument("--out", default="baselines/baseline.json")
        if name == "export-preferences":
            p.add_argument("--out", default="results/preferences.jsonl")
    k = sub.add_parser("kappa")
    k.add_argument("--labels", default=LABELS)
    args = ap.parse_args(argv)

    if args.cmd == "kappa":
        labels = read_jsonl(args.labels)
        anns = sorted({r["annotator"] for r in labels})
        ids, a, b = align(labels, anns[0], anns[1])
        kap = cohen_kappa(a, b)
        print(json.dumps({"raters": anns[:2], "n": len(ids), "kappa": round(kap, 4),
                          "interpretation": interpret_kappa(kap)}))
        return 0

    from .judge import get_judge

    items = read_jsonl(args.data)
    labels = read_jsonl(args.labels) if args.labels else None
    result = run(items, labels, get_judge(args.judge), tuple(args.systems.split(",")))
    if args.cmd == "run":
        save(result, args.out)
        write_jsonl(args.rows, [r | {"system": s} for s, rows in result["rows"].items() for r in rows])
        md = to_markdown(result, args.data)
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
        print(md)
    elif args.cmd == "export-preferences":
        write_jsonl(args.out, result.get("preference_pairs", []))
        print(f"wrote {len(result.get('preference_pairs', []))} preference pairs to {args.out}")
    elif args.cmd == "baseline":
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(make_baseline(result), f, indent=2)
            f.write("\n")
        print(f"baseline written to {args.out}")
    elif args.cmd == "check":
        with open(args.baseline, encoding="utf-8") as f:
            failures = check_baseline(result, json.load(f), args.tolerance)
        if failures:
            print("REGRESSION:\n  " + "\n  ".join(failures))
            return 1
        print("OK: no metric below baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
