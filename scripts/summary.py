# usage: python scripts/summary.py
# Single command that prints every table and test the README references.
# Useful for reviewers checking that the numbers in the writeup come
# straight from the data files.
import json
from itertools import combinations
from math import comb, erfc, sqrt
from pathlib import Path

from eval.report import wilson, cross_table

RESULTS = Path("results")


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    n1 = a + b
    n2 = c + d
    total = n1 + n2
    tot_pos = a + c
    if tot_pos == 0 or tot_pos == total or n1 == 0 or n2 == 0:
        return 1.0
    p_obs = comb(n1, a) * comb(n2, c) / comb(total, tot_pos)
    p_total = 0.0
    for ai in range(max(0, tot_pos - n2), min(n1, tot_pos) + 1):
        ci = tot_pos - ai
        p_i = comb(n1, ai) * comb(n2, ci) / comb(total, tot_pos)
        if p_i <= p_obs + 1e-12:
            p_total += p_i
    return p_total


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    chi2 = (abs(b - c) - 1) ** 2 / n
    return erfc(sqrt(chi2) / sqrt(2))


def holm(pvals: list[tuple[str, float]]) -> dict[str, float]:
    sorted_p = sorted(enumerate(pvals), key=lambda x: x[1][1])
    m = len(pvals)
    adj: list[float] = [0.0] * m
    running = 0.0
    for rank, (i, (_, p)) in enumerate(sorted_p):
        a = min(1.0, max(running, (m - rank) * p))
        adj[i] = a
        running = a
    return {label: adj[i] for i, (label, _) in enumerate(pvals)}


def hint_sweep_report(tag: str = "mellum_sft") -> str:
    out = ["", "## Hint-format sweep (paired, n=138, T=0.6 retries)"]
    formats = {
        "current":   RESULTS / f"{tag}_multiturn.jsonl",
        "minimal":   RESULTS / f"{tag}_hintsweep_minimal.jsonl",
        "traceback": RESULTS / f"{tag}_hintsweep_traceback.jsonl",
        "post":      RESULTS / f"{tag}_hintsweep_post.jsonl",
        "no_hint":   RESULTS / f"{tag}_multiturn_nohint.jsonl",
    }
    available = {}
    for f, p in formats.items():
        if p.exists():
            rows = [json.loads(l) for l in p.open() if l.strip()]
            available[f] = {r["task_id"]: r["final_plus_passed"]
                            for r in rows if not r.get("skipped_reason")}
    if "no_hint" not in available:
        return "\n".join(out + ["  no-hint baseline missing; skip"])
    common = set.intersection(*(set(d) for d in available.values()))
    for f in ("no_hint", "minimal", "traceback", "current", "post"):
        if f not in available:
            continue
        d = available[f]
        k = sum(d[t] for t in common)
        p, lo, hi = wilson(k, len(common))
        out.append(f"  {f:<10} {k}/{len(common)} = {p:.1%}  [{lo:.1%}, {hi:.1%}]")
    out.append("\n  Confirmatory: 4 contrasts vs no_hint, Holm-Bonferroni:")
    nh = available["no_hint"]
    raw = []
    contr = []
    for f in ("minimal", "traceback", "current", "post"):
        if f not in available:
            continue
        d = available[f]
        bb = sum(1 for t in common if d[t] and not nh[t])
        cc = sum(1 for t in common if not d[t] and nh[t])
        p = mcnemar(bb, cc)
        raw.append((f, p))
        contr.append((f, bb, cc, p))
    adj = holm(raw)
    for f, bb, cc, p in contr:
        sig = " **" if adj[f] < 0.01 else (" *" if adj[f] < 0.05 else "")
        out.append(f"    {f:<10} vs no_hint  b={bb:3d} c={cc:3d}  p={p:.3f}  p_holm={adj[f]:.3f}{sig}")
    return "\n".join(out)


def regression_cross_family() -> str:
    rows = []
    for tag in ("mellum_sft", "mellum_dpo", "ds_base", "ds_instruct"):
        p = RESULTS / f"{tag}_regression.jsonl"
        if not p.exists():
            continue
        data = [json.loads(l) for l in p.open() if l.strip()]
        broken = sum(1 for r in data if not r["still_passes"])
        rows.append((tag, broken, len(data)))
    if len(rows) < 2:
        return ""
    out = ["", "## Cross-family regression rate (Fisher's two-sided exact)"]
    out.append(f"  {'tag':<18}{'rate':>20}")
    for tag, k, n in rows:
        p, lo, hi = wilson(k, n)
        out.append(f"  {tag:<18} {k}/{n} = {p:.1%}  [{lo:.1%}, {hi:.1%}]")
    out.append("")
    # Confirmatory family: the 4 cross-family Mellum-vs-DeepSeek contrasts.
    # Within-family pairs (Mellum-SFT vs DPO, DS-base vs instruct) are
    # exploratory and not corrected with the cross-family family.
    by_tag = {tag: (k, n) for tag, k, n in rows}
    mellum = [t for t in ("mellum_sft", "mellum_dpo") if t in by_tag]
    deepseek = [t for t in ("ds_base", "ds_instruct") if t in by_tag]
    confirmatory = []
    for m in mellum:
        for d in deepseek:
            km, nm = by_tag[m]
            kd, nd = by_tag[d]
            p = fisher_exact(km, nm - km, kd, nd - kd)
            confirmatory.append((f"{m} vs {d}", p))
    if confirmatory:
        out.append("  Confirmatory: cross-family contrasts, Holm-Bonferroni:")
        adj = holm(confirmatory)
        for label, p in confirmatory:
            ap = adj[label]
            sig = " **" if ap < 0.01 else (" *" if ap < 0.05 else "")
            out.append(f"    {label:<32}  p={p:.4f}  p_holm={ap:.4f}{sig}")
    out.append("\n  Exploratory (within-family, uncorrected):")
    for fam in (mellum, deepseek):
        if len(fam) >= 2:
            ka, na = by_tag[fam[0]]
            kb, nb = by_tag[fam[1]]
            p = fisher_exact(ka, na - ka, kb, nb - kb)
            out.append(f"    {fam[0]} vs {fam[1]:<20}  p={p:.4f}")
    return "\n".join(out)


def main() -> None:
    print("# coding-model-eval summary\n")
    print("## Cross-table")
    print(cross_table())
    print(hint_sweep_report("mellum_sft"))
    print(regression_cross_family())


if __name__ == "__main__":
    main()
