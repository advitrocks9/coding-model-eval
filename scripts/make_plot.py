"""Make the recovery-vs-regression scatter for the writeup.

X axis: regression rate (% of correct solutions broken by a synthetic
'try again' hint). Y axis: recovery rate (% of failed solutions recovered
under the same hint format). One point per (model, hint_format).

If a point is in the upper-left, the hint is helping. If lower-right,
the hint is doing more harm than good. The Mellum-SFT and Mellum-DPO
points with the short hint format both sit below the y=x diagonal,
which is the finding the writeup turns on.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = Path("results")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def recovery_rate(multi_path: Path) -> tuple[float, int, int]:
    rows = load(multi_path)
    retried = [r for r in rows if not r.get("skipped_reason")]
    recovered = [r for r in retried if r["final_plus_passed"]]
    return len(recovered) / len(retried), len(recovered), len(retried)


def regression_rate(reg_path: Path) -> tuple[float, int, int]:
    rows = load(reg_path)
    broken = sum(1 for r in rows if not r["still_passes"])
    return broken / len(rows), broken, len(rows)


def main() -> None:
    points = []
    for tag, label in [
        ("mellum_sft", "Mellum-4b-sft-python"),
        ("mellum_dpo", "Mellum-4b-dpo-python"),
    ]:
        mt = RESULTS / f"{tag}_multiturn.jsonl"
        reg = RESULTS / f"{tag}_regression.jsonl"
        if not (mt.exists() and reg.exists()):
            continue
        rec, _, _ = recovery_rate(mt)
        regr, _, _ = regression_rate(reg)
        points.append((label + " (short hint)", regr, rec))

    nohint_path = RESULTS / "mellum_sft_multiturn_nohint.jsonl"
    if nohint_path.exists():
        rec, _, _ = recovery_rate(nohint_path)
        # no regression test for the no-hint case; placeholder x=0 since
        # there's no hint to break correct solutions
        points.append(("Mellum-SFT (no hint, sampled)", 0.0, rec))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.scatter(xs, ys, s=80, color="#1f77b4")
    for label, x, y in points:
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    lim = max(max(xs, default=0), max(ys, default=0)) * 1.2 + 0.05
    ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=1, alpha=0.5)
    ax.set_xlabel("regression rate  (correct broken by retry hint)")
    ax.set_ylabel("recovery rate  (failed fixed by retry hint)")
    ax.set_title("Multi-turn correction is below the diagonal for Mellum")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.005, lim)
    ax.grid(alpha=0.3)
    out = Path("assets/recovery_vs_regression.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for label, x, y in points:
        print(f"  {label}: regression={x:.1%} recovery={y:.1%}")


if __name__ == "__main__":
    main()
