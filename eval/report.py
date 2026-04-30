from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

RESULTS = Path("results")
Z95 = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = Z95
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _codex_passk(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def all_tags() -> list[str]:
    if not RESULTS.exists():
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for p in sorted(RESULTS.glob("*_singleturn.jsonl")):
        t = p.stem.removesuffix("_singleturn")
        if t not in seen:
            tags.append(t)
            seen.add(t)
    for p in sorted(RESULTS.glob("*_fim.jsonl")):
        t = p.stem
        if t not in seen:
            tags.append(t)
            seen.add(t)
    return tags


def _fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{p:.1%} [{lo:.1%}, {hi:.1%}]"


def cross_table() -> str:
    tags = all_tags()
    head = (
        f"{'tag':<26}{'base':>22}{'plus':>22}"
        f"{'+retry':>22}{'recover':>10}{'regress':>13}"
    )
    lines = [head, "-" * len(head)]
    for tag in tags:
        if tag.endswith("_fim"):
            src = RESULTS / f"{tag}.jsonl"
            mt = reg = None
        else:
            src = RESULTS / f"{tag}_singleturn.jsonl"
            mt = RESULTS / f"{tag}_multiturn.jsonl"
            reg = RESULTS / f"{tag}_regression.jsonl"

        if src and src.exists():
            rows = _load(src)
            n = len(rows)
            base_k = sum(r["base_pass"] for r in rows)
            plus_k = sum(r["plus_pass"] for r in rows)
            base_s = _fmt_ci(base_k, n)
            plus_s = _fmt_ci(plus_k, n)
        else:
            base_s = plus_s = "-"

        if mt and mt.exists():
            mrows = _load(mt)
            final = sum(r["final_plus_passed"] for r in mrows)
            retried = [r for r in mrows if not r.get("skipped_reason")]
            recovered = [r for r in retried if r["final_plus_passed"]]
            retry_s = _fmt_ci(final, len(mrows))
            recov_s = f"{len(recovered)}/{len(retried)}" if retried else "-"
        else:
            retry_s = recov_s = "-"

        if reg and reg.exists():
            rrows = _load(reg)
            broken = sum(1 for x in rrows if not x["still_passes"])
            reg_s = f"{broken}/{len(rrows)}" if rrows else "-"
        else:
            reg_s = "-"

        lines.append(
            f"{tag:<26}{base_s:>22}{plus_s:>22}"
            f"{retry_s:>22}{recov_s:>10}{reg_s:>13}"
        )
    return "\n".join(lines)


def detail(tag: str) -> str:
    out: list[str] = [f"# {tag}"]
    s = RESULTS / f"{tag}_singleturn.jsonl"
    f = RESULTS / f"{tag}_fim.jsonl"
    src = s if s.exists() else f
    if src.exists():
        rows = _load(src)
        n = len(rows)
        base_k = sum(r["base_pass"] for r in rows)
        plus_k = sum(r["plus_pass"] for r in rows)
        out.append(f"  HumanEval base = {base_k}/{n} = {_fmt_ci(base_k, n)}")
        out.append(f"  HumanEval+    = {plus_k}/{n} = {_fmt_ci(plus_k, n)}")
        gap = [r for r in rows if r["base_pass"] and not r["plus_pass"]]
        if gap:
            out.append(f"  base-pass / plus-fail = {len(gap)}")
            kinds = Counter(r["plus_kind"] for r in gap)
            for k, v in kinds.most_common():
                out.append(f"    {k}: {v}")

    m = RESULTS / f"{tag}_multiturn.jsonl"
    if m.exists():
        rows = _load(m)
        retried = [r for r in rows if not r.get("skipped_reason")]
        recov = [r for r in retried if r["final_plus_passed"]]
        out.append(
            f"  multi-turn recovery (with hint) = "
            f"{len(recov)}/{len(retried)} = {_fmt_ci(len(recov), len(retried))}"
        )
        if recov:
            turns = Counter(r["recovered_at"] for r in recov)
            out.append(f"  recovered at turn: {dict(sorted(turns.items()))}")

    r = RESULTS / f"{tag}_regression.jsonl"
    if r.exists():
        rows = _load(r)
        broken = sum(1 for x in rows if not x["still_passes"])
        out.append(
            f"  regression rate = {broken}/{len(rows)} = "
            f"{_fmt_ci(broken, len(rows))}"
        )

    n_path = RESULTS / f"{tag}_multiturn_nohint.jsonl"
    if n_path.exists():
        rows = _load(n_path)
        retried = [r for r in rows if not r.get("skipped_reason")]
        recov = [r for r in retried if r["final_plus_passed"]]
        out.append(
            f"  no-hint recovery = "
            f"{len(recov)}/{len(retried)} = {_fmt_ci(len(recov), len(retried))}"
        )

    for fmt in ("minimal", "traceback", "post"):
        s_path = RESULTS / f"{tag}_hintsweep_{fmt}.jsonl"
        if s_path.exists():
            rows = _load(s_path)
            retried = [r for r in rows if not r.get("skipped_reason")]
            recov = [r for r in retried if r["final_plus_passed"]]
            out.append(
                f"  hint={fmt:<10} recovery = "
                f"{len(recov)}/{len(retried)} = {_fmt_ci(len(recov), len(retried))}"
            )

    pk_path = RESULTS / f"{tag}_passk.jsonl"
    if pk_path.exists():
        rows = _load(pk_path)
        n = len(rows)
        n_samples = rows[0]["n_samples"]
        T = rows[0]["temperature"]
        for k_val in (1, 2, 4):
            if k_val > n_samples:
                continue
            unbiased = sum(_codex_passk(n_samples, r["correct_count"], k_val) for r in rows) / n
            any_pass = sum(1 for r in rows if r["correct_count"] >= 1)
            out.append(
                f"  pass@{k_val} (sampled, T={T}, n={n_samples}) = "
                f"{unbiased:.1%} (unbiased)"
            )
        out.append(f"  any-of-{n_samples} = {any_pass}/{n} = {_fmt_ci(any_pass, n)}")

    return "\n".join(out)


def main() -> None:
    if len(sys.argv) > 1:
        for tag in sys.argv[1:]:
            print(detail(tag))
            print()
    else:
        print(cross_table())
        print()
        for tag in all_tags():
            print(detail(tag))
            print()


if __name__ == "__main__":
    main()
