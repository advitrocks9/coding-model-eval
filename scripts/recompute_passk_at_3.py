# usage: python scripts/recompute_passk_at_3.py [TAG]
# The README sold the compute-matched comparator as "budget = 3 generations"
# but `scripts/run_passk.py` defaults to 4 and the committed JSONL has
# n_samples=4. Recompute pass@k at n=3 from the first 3 logged samples per
# task. Codex-style unbiased estimator: 1 - C(n-c, k) / C(n, k).
import json
import math
import sys
from pathlib import Path


def codex_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "mellum_sft"
    src = Path(f"results/{tag}_passk.jsonl")
    dst = Path(f"results/{tag}_passk_at_3.jsonl")

    rows = [json.loads(l) for l in src.open()]
    out = dst.open("w")
    p1_terms: list[float] = []
    p2_terms: list[float] = []
    p3_terms: list[float] = []
    n_truncated = 0
    for r in rows:
        samples = r["samples"][:3]
        if len(r["samples"]) > 3:
            n_truncated += 1
        n = len(samples)
        c = sum(1 for s in samples if s["plus_pass"])
        p1_terms.append(codex_pass_at_k(n, c, 1))
        p2_terms.append(codex_pass_at_k(n, c, 2))
        p3_terms.append(codex_pass_at_k(n, c, 3))
        out.write(json.dumps({
            "task_id": r["task_id"],
            "n_samples": n,
            "temperature": r["temperature"],
            "correct_count": c,
            "pass_at_1": c >= 1,
            "pass_at_2": c >= 2,
            "pass_at_3": c >= 3,
            "samples": samples,
        }) + "\n")
    out.close()

    n_tasks = len(rows)
    print(f"truncated {n_truncated} / {n_tasks} tasks from 4 -> 3 samples")
    print(f"source: {src}")
    print(f"out:    {dst}")
    print(f"pass@1 (unbiased) = {sum(p1_terms)/n_tasks:.3%}")
    print(f"pass@2 (unbiased) = {sum(p2_terms)/n_tasks:.3%}")
    print(f"pass@3 (unbiased) = {sum(p3_terms)/n_tasks:.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
