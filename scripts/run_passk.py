# usage: python scripts/run_passk.py [MODEL_PATH] [TAG] [N_SAMPLES] [TEMPERATURE]
import json
import math
import sys
import time
from pathlib import Path

from tqdm import tqdm

from eval.loaders import load_tasks
from eval.multi_turn import _stable_seed
from eval.runner import Generator
from eval.sandbox import execute

DEFAULT_MODEL = "JetBrains/Mellum-4b-sft-python"
DEFAULT_TAG = "mellum_sft"
DEFAULT_N = 4
DEFAULT_T = 0.6


def codex_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_T
    out_path = Path(f"results/{tag}_passk.jsonl")

    tasks = load_tasks()
    print(f"loaded {len(tasks)} tasks", flush=True)
    print(f"model: {model_path}\ntag: {tag}\nN={n_samples}, T={temperature}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = Generator(model_path)
    print("model loaded", flush=True)

    out = out_path.open("w")
    t0 = time.time()
    correct_counts: list[int] = []
    for t in tqdm(tasks):
        seed_base = _stable_seed(t.task_id)
        per_sample: list[dict] = []
        c = 0
        for s in range(n_samples):
            completion = g.complete(t.prompt, temperature=temperature, seed=seed_base + s)
            full = t.prompt + completion
            r = execute(full, t.test_plus + f"\ncheck({t.entry_point})\n", timeout=20)
            per_sample.append({
                "sample": s,
                "completion": completion,
                "plus_pass": r.passed,
                "plus_kind": r.error_kind,
            })
            if r.passed:
                c += 1
        correct_counts.append(c)
        out.write(json.dumps({
            "task_id": t.task_id,
            "n_samples": n_samples,
            "temperature": temperature,
            "correct_count": c,
            "pass_at_1": c >= 1,
            "pass_at_2": c >= 2 if n_samples >= 2 else None,
            "pass_at_4": c >= 1 and n_samples >= 4,
            "samples": per_sample,
        }) + "\n")
        out.flush()
    out.close()

    n = len(tasks)
    p1 = sum(codex_pass_at_k(n_samples, c, 1) for c in correct_counts) / n
    print(f"\npass@1 (unbiased estimator) = {p1:.3%}")
    if n_samples >= 2:
        p2 = sum(codex_pass_at_k(n_samples, c, 2) for c in correct_counts) / n
        print(f"pass@2 (unbiased estimator) = {p2:.3%}")
    if n_samples >= 4:
        p4 = sum(codex_pass_at_k(n_samples, c, 4) for c in correct_counts) / n
        print(f"pass@4 (unbiased estimator) = {p4:.3%}")
    dt = time.time() - t0
    print(f"wallclock {dt:.0f}s, {dt/n:.1f}s/task, {dt/(n*n_samples):.1f}s/sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
