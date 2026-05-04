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
    """Exact two-sided McNemar p-value for paired discordant counts (b, c).

    b: format A passes, format B fails.
    c: format A fails, format B passes.
    Under H0 (no difference), each discordant pair flips a fair coin, so
    min(b, c) follows Binomial(b+c, 0.5). The exact tail is tractable for
    any n we'll see here (largest is ~20 discordant pairs), so always
    return the exact binomial p; chi2 is left as NaN since the
    normal-approx isn't what's reported.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float("nan"), min(1.0, 2 * p)


def load_no_hint() -> dict[str, bool]:
    path = RESULTS / f"{TAG}_multiturn_nohint.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return {r["task_id"]: r["final_plus_passed"]
            for r in rows if not r.get("skipped_reason")}


def holm(pvals: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
    # holm-bonferroni step-down. returns (label, raw_p, adjusted_p).
    sorted_p = sorted(enumerate(pvals), key=lambda x: x[1][1])
    m = len(pvals)
    adj: list[float | None] = [None] * m
    running = 0.0
    for rank, (orig_i, (_, p)) in enumerate(sorted_p):
        adj_i = min(1.0, max(running, (m - rank) * p))
        adj[orig_i] = adj_i
        running = adj_i
    return [(label, p, adj[i]) for i, (label, p) in enumerate(pvals)]


def main() -> None:
    formats = ["minimal", "current", "traceback", "post"]
    available = {f: load_format(f) for f in formats}
    available = {f: d for f, d in available.items() if d}
    print(f"available formats: {list(available)}")

    no_hint = load_no_hint()
    if no_hint:
        available["no_hint"] = no_hint

    common = set.intersection(*(set(d) for d in available.values())) if available else set()
    print(f"tasks common across formats: {len(common)}\n")

    print(f"{'format':<12}{'recovery':>30}")
    for f, d in sorted(available.items(), key=lambda x: -sum(x[1][t] for t in common)):
        passes = sum(d[t] for t in common)
        p, lo, hi = wilson(passes, len(common))
        print(f"  {f:<10} {passes}/{len(common)} = {p:.1%}  [{lo:.1%}, {hi:.1%}]")

    if "no_hint" in available:
        print("\nConfirmatory family: 4 paired McNemar contrasts vs no-hint (Holm-Bonferroni).")
        nh = available["no_hint"]
        contrasts = []
        raw_ps = []
        for f in ("minimal", "traceback", "current", "post"):
            if f not in available:
                continue
            df = available[f]
            bb = sum(1 for t in common if df[t] and not nh[t])
            cc = sum(1 for t in common if not df[t] and nh[t])
            chi2, p = mcnemar(bb, cc)
            contrasts.append((f"{f} vs no_hint", bb, cc, chi2, p))
            raw_ps.append((f, p))
        adjusted = holm(raw_ps)
        adj_lookup = {label: adj_p for label, _, adj_p in adjusted}
        for label_b, bb, cc, chi2, raw_p in contrasts:
            f = label_b.split()[0]
            adj_p = adj_lookup[f]
            sig = "  **" if adj_p < 0.01 else ("  *" if adj_p < 0.05 else "")
            print(f"  {label_b:<28} b={bb:3d} c={cc:3d}  p={raw_p:.3f}  p_holm={adj_p:.3f}{sig}")

    print("\nExploratory pairwise (uncorrected; secondary):")
    for a, b in combinations(available, 2):
        if "no_hint" in (a, b):
            continue
        da, db = available[a], available[b]
        bb = sum(1 for t in common if da[t] and not db[t])
        cc = sum(1 for t in common if not da[t] and db[t])
        _, p = mcnemar(bb, cc)
        print(f"  {a} vs {b}: b={bb}, c={cc}, p={p:.3f}")


if __name__ == "__main__":
    main()
