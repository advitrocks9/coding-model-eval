# usage: python scripts/analyze_hint_sweep.py [TAG]
# paired McNemar across hint formats. tells you whether two formats
# differ by more than sampling noise on the SAME tasks.
import json
import sys
from itertools import combinations
from pathlib import Path

from eval.report import wilson


TAG = sys.argv[1] if len(sys.argv) > 1 else "mellum_sft"
RESULTS = Path("results")


def load_format(fmt: str) -> dict[str, bool]:
    if fmt == "current":
        # the original multi-turn run is the "current" format baseline.
        # but it's run on all 164 tasks (with skips); reduce to the same
        # set of failed tasks the sweep covers.
        path = RESULTS / f"{TAG}_multiturn.jsonl"
    else:
        path = RESULTS / f"{TAG}_hintsweep_{fmt}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return {
        r["task_id"]: r["final_plus_passed"]
        for r in rows
        if not r.get("skipped_reason")
    }


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Return (chi2, p_two_sided) for paired discordant pairs (b, c).

    b: format A passes, format B fails.
    c: format A fails, format B passes.
    Continuity-corrected. Approximate normal for n >= 25; for tiny n
    use the binomial form.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < 25:
        # exact binomial test, two-sided
        from math import comb
        k = min(b, c)
        p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        return float(b - c), min(1.0, 2 * p)
    chi2 = (abs(b - c) - 1) ** 2 / n
    # rough normal-approx p; correct enough for n we care about
    from math import erfc, sqrt
    z = sqrt(chi2)
    p = erfc(z / sqrt(2))
    return chi2, p


def main() -> None:
    formats = ["minimal", "current", "traceback", "post"]
    available = {f: load_format(f) for f in formats}
    # only formats with data
    available = {f: d for f, d in available.items() if d}
    print(f"available formats: {list(available)}")

    common = set.intersection(*(set(d) for d in available.values())) if available else set()
    print(f"tasks common across formats: {len(common)}\n")

    print(f"{'format':<12}{'recovery':>20}")
    for f, d in available.items():
        passes = sum(d[t] for t in common)
        print(f"{f:<12}{passes}/{len(common)}  {wilson(passes, len(common))[0]:.1%} "
              f"[{wilson(passes, len(common))[1]:.1%}, {wilson(passes, len(common))[2]:.1%}]")

    print(f"\nMcNemar paired tests (b = A passes & B fails, c = A fails & B passes):")
    for a, b in combinations(available, 2):
        da, db = available[a], available[b]
        bb = sum(1 for t in common if da[t] and not db[t])
        cc = sum(1 for t in common if not da[t] and db[t])
        chi2, p = mcnemar(bb, cc)
        sig = "  ***" if p < 0.01 else ("  *" if p < 0.05 else "")
        print(f"  {a} vs {b}: b={bb}, c={cc}, chi2={chi2:.2f}, p={p:.3f}{sig}")


if __name__ == "__main__":
    main()
