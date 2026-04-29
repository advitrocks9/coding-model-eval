"""Aggregation and printing for results files.

The two artefacts:
    summary_table(): one-row-per-experiment, base/plus/multi pass@1
    failure_breakdown(): what kind of failure (assertion, exception, timeout)
                         and what multi-turn recovered

Run this last, when the JSONL files are populated.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def summary_table(single_path: Path, multi_path: Path | None = None) -> str:
    s = _load(single_path)
    n = len(s)
    base = sum(r["base_pass"] for r in s) / n
    plus = sum(r["plus_pass"] for r in s) / n

    rows = [
        ("HumanEval (base tests)", f"{base:.1%}", f"{int(base*n)}/{n}"),
        ("HumanEval+ (EvalPlus tests)", f"{plus:.1%}", f"{int(plus*n)}/{n}"),
        ("Penalty (base - plus)", f"{base - plus:+.1%}", "-"),
    ]
    if multi_path is not None and multi_path.exists():
        m = _load(multi_path)
        # final_plus_passed includes already-passed reuses
        final = sum(r["final_plus_passed"] for r in m) / n
        retried = [r for r in m if r.get("skipped_reason") != "already_passed"]
        recovered = [r for r in retried if r["final_plus_passed"]]
        recovered_n = len(recovered)
        retried_n = len(retried)

        rows.append(("HumanEval+ after up to 2 retries", f"{final:.1%}", f"{int(final*n)}/{n}"))
        rows.append((
            f"  recovered by retry",
            f"{recovered_n/retried_n:.1%}" if retried_n else "-",
            f"{recovered_n}/{retried_n}",
        ))

    width = max(len(r[0]) for r in rows) + 2
    out = []
    out.append(f"{'metric':<{width}}{'pass@1':>10}{'count':>14}")
    out.append("-" * (width + 24))
    for r in rows:
        out.append(f"{r[0]:<{width}}{r[1]:>10}{r[2]:>14}")
    return "\n".join(out)


def failure_breakdown(single_path: Path) -> str:
    """Where the EvalPlus penalty comes from, by failure kind."""
    s = _load(single_path)
    base_pass_plus_fail = [r for r in s if r["base_pass"] and not r["plus_pass"]]
    cnt = Counter(r["plus_kind"] for r in base_pass_plus_fail)
    out = [f"tasks that pass base tests but fail EvalPlus tests: {len(base_pass_plus_fail)}"]
    out.append("by failure kind on EvalPlus:")
    for kind, n in cnt.most_common():
        out.append(f"  {kind:<10} {n}")
    return "\n".join(out)


def recovery_breakdown(multi_path: Path) -> str:
    """For each task that multi-turn recovered, which turn fixed it?"""
    if not multi_path.exists():
        return "(multi-turn results not yet available)"
    m = _load(multi_path)
    retried = [r for r in m if r.get("skipped_reason") != "already_passed"]
    recovered = [r for r in retried if r["final_plus_passed"]]
    out = [f"recovered tasks: {len(recovered)} / {len(retried)} retried"]
    by_turn = Counter(r["recovered_at"] for r in recovered)
    out.append("recovery turn distribution (turn 0 = first retry's prompt):")
    for k in sorted(by_turn):
        out.append(f"  turn {k}: {by_turn[k]}")
    # the failure kinds that did NOT recover
    not_recovered = [r for r in retried if not r["final_plus_passed"]]
    last_kinds = Counter(r["turns"][-1]["plus_kind"] for r in not_recovered)
    out.append("\nfailure kind on tasks multi-turn could not fix:")
    for k, n in last_kinds.most_common():
        out.append(f"  {k:<10} {n}")
    return "\n".join(out)


if __name__ == "__main__":
    s = Path("results/mellum_sft_singleturn.jsonl")
    m = Path("results/mellum_sft_multiturn.jsonl")
    print(summary_table(s, m))
    print()
    print(failure_breakdown(s))
    print()
    print(recovery_breakdown(m))
