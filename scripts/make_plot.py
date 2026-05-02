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
    hint_points = []
    nohint_points = []
    for tag, label in [
        ("mellum_sft", "Mellum-SFT"),
        ("mellum_dpo", "Mellum-DPO"),
        ("ds_base", "DS-base"),
        ("ds_instruct", "DS-instruct"),
    ]:
        mt = RESULTS / f"{tag}_multiturn.jsonl"
        reg = RESULTS / f"{tag}_regression.jsonl"
        if mt.exists() and reg.exists():
            rec, _, _ = recovery_rate(mt)
            regr, _, _ = regression_rate(reg)
            hint_points.append((label, regr, rec))
        nohint = RESULTS / f"{tag}_multiturn_nohint.jsonl"
        if nohint.exists():
            rec, _, _ = recovery_rate(nohint)
            nohint_points.append((label + " (no hint)", 0.0, rec))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    if hint_points:
        xs = [p[1] for p in hint_points]
        ys = [p[2] for p in hint_points]
        ax.scatter(xs, ys, s=80, color="#1f77b4", label="current hint")
        for label, x, y in hint_points:
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(8, 6), fontsize=9)
    if nohint_points:
        xs2 = [p[1] for p in nohint_points]
        ys2 = [p[2] for p in nohint_points]
        ax.scatter(xs2, ys2, s=80, color="#d62728", marker="^", label="no hint")
        for label, x, y in nohint_points:
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(8, 6), fontsize=9)
    all_xs = [p[1] for p in hint_points] + [p[1] for p in nohint_points]
    all_ys = [p[2] for p in hint_points] + [p[2] for p in nohint_points]
    lim = max(max(all_xs, default=0), max(ys, default=0)) * 1.2 + 0.05
    ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=1, alpha=0.5)
    ax.set_xlabel("regression rate  (correct broken by retry hint)")
    ax.set_ylabel("recovery rate  (failed fixed by retry hint)")
    ax.set_title("Recovery vs regression across 4 post-trainings, with/without hint")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.005, max(all_ys, default=0) * 1.2 + 0.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    out = Path("assets/recovery_vs_regression.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for label, x, y in hint_points + nohint_points:
        print(f"  {label}: regression={x:.1%} recovery={y:.1%}")


if __name__ == "__main__":
    main()
