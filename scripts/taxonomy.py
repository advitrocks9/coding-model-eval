# usage: python scripts/taxonomy.py [TAG]
# categorise multi-turn failures by failure mode. uses sandbox kind +
# completion string heuristics. no hand labelling; the categories that
# need them are flagged.
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


TAG = sys.argv[1] if len(sys.argv) > 1 else "mellum_sft"
RESULTS = Path("results")


def classify_completion(completion: str) -> str:
    body = completion.strip()
    if not body or body in ("pass", "..."):
        return "gave_up"
    lines = [l for l in body.splitlines() if l.strip()]
    code_lines = [l for l in lines if not l.strip().startswith("#")]
    if not code_lines:
        return "all_comments"
    if len(code_lines) == 1 and code_lines[0].strip() == "pass":
        return "gave_up"
    return "wrote_code"


def kind_to_category(kind: str, completion: str) -> str:
    if kind == "timeout":
        return "timeout"
    if kind == "syntax":
        return "syntax_error"
    if kind == "exception":
        return "exception"
    sub = classify_completion(completion)
    if sub == "gave_up":
        return "gave_up"
    if sub == "all_comments":
        return "all_comments"
    return "logic_wrong"


def report(name: str, path: Path) -> None:
    if not path.exists():
        print(f"{name}: no data")
        return
    rows = [json.loads(l) for l in path.open() if l.strip()]
    retried = [r for r in rows if not r.get("skipped_reason")]
    by_outcome: dict[str, list[dict]] = defaultdict(list)
    for r in retried:
        last_turn = r["turns"][-1]
        cat = kind_to_category(last_turn["plus_kind"], last_turn["completion"])
        outcome = "recovered" if r["final_plus_passed"] else "stuck"
        by_outcome[outcome].append((cat, r["task_id"]))

    print(f"\n=== {name} ({len(retried)} retried) ===")
    for outcome in ("recovered", "stuck"):
        items = by_outcome[outcome]
        if not items:
            continue
        cats = Counter(cat for cat, _ in items)
        print(f"  {outcome}: {len(items)}")
        for cat, n in cats.most_common():
            print(f"    {cat:<14} {n}")


def main() -> None:
    print(f"failure taxonomy for tag={TAG}")
    report("with-hint multi-turn", RESULTS / f"{TAG}_multiturn.jsonl")
    report("no-hint multi-turn", RESULTS / f"{TAG}_multiturn_nohint.jsonl")
    for fmt in ("minimal", "current", "traceback", "post"):
        p = RESULTS / f"{TAG}_hintsweep_{fmt}.jsonl"
        if p.exists():
            report(f"hint-sweep ({fmt})", p)


if __name__ == "__main__":
    main()
