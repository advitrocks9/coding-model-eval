# usage: python scripts/calibrate_temperature.py [MODEL_PATH]
import json
import sys
import time
from pathlib import Path

from eval.loaders import load_tasks
from eval.runner import Generator
from eval.sandbox import execute

DEFAULT_MODEL = "/home/prannayk/models/mellum-sft-python"
TEMPS = (0.2, 0.6, 1.0)
N_SAMPLES = 4
SLICE = [
    "HumanEval/0",
    "HumanEval/2",
    "HumanEval/4",
    "HumanEval/13",
    "HumanEval/22",
    "HumanEval/35",
    "HumanEval/79",
    "HumanEval/53",
]


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    out_path = Path("results/calibration_temperature.jsonl")
    tasks_all = {t.task_id: t for t in load_tasks()}
    tasks = [tasks_all[tid] for tid in SLICE]

    g = Generator(model_path)
    print(f"calibrating T on {len(tasks)} tasks, {N_SAMPLES} samples each, T in {TEMPS}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("w")
    summary: dict[float, list[int]] = {t: [] for t in TEMPS}

    t0 = time.time()
    for T in TEMPS:
        print(f"\n--- T = {T} ---", flush=True)
        for t in tasks:
            seed_base = abs(hash(t.task_id)) % 100_000
            c = 0
            for s in range(N_SAMPLES):
                completion = g.complete(t.prompt, temperature=T, seed=seed_base + s)
                full = t.prompt + completion
                r = execute(full, t.test_plus + f"\ncheck({t.entry_point})\n", timeout=20)
                if r.passed:
                    c += 1
            summary[T].append(c)
            print(f"  {t.task_id}: {c}/{N_SAMPLES}", flush=True)
            out.write(json.dumps({
                "task_id": t.task_id,
                "temperature": T,
                "n_samples": N_SAMPLES,
                "correct_count": c,
            }) + "\n")
            out.flush()
    out.close()

    print(f"\nwallclock {time.time() - t0:.0f}s\n")
    print("T     pass@1  pass@4  total_correct/total")
    for T, counts in summary.items():
        n = len(counts) * N_SAMPLES
        total_c = sum(counts)
        p4 = sum(1 for c in counts if c >= 1) / len(counts)
        rate = total_c / n
        print(f"{T:<6}{rate:.1%}   {p4:.1%}   {total_c}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
