# usage: python scripts/run_humaneval_infilling.py [MODEL_PATH] [TAG] [VARIANT]
# variant in {single, multi, random, light}; default single (1033 tasks).
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from eval.fim_loaders import load_fim_tasks
from eval.runner import Generator
from eval.sandbox import execute

DEFAULT_MODEL = "/home/prannayk/models/mellum-base"
DEFAULT_TAG = "mellum_base"
DEFAULT_VARIANT = "single"


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    variant = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_VARIANT
    out_path = Path(f"results/{tag}_he_infill_{variant}.jsonl")

    tasks = load_fim_tasks(variant)
    print(f"loaded {len(tasks)} {variant}-line FIM tasks", flush=True)
    print(f"model: {model_path}\ntag: {tag}\nout: {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = Generator(model_path)
    print("model loaded", flush=True)

    out = out_path.open("w")
    passed = 0
    t0 = time.time()
    for t in tqdm(tasks):
        completion = g.fim_complete(t.prompt, suffix=t.suffix, filename=f"{t.entry_point}.py")
        full = t.prompt + completion + t.suffix
        r = execute(full, t.test + f"\ncheck({t.entry_point})\n", timeout=20)
        passed += r.passed
        out.write(json.dumps({
            "task_id": t.task_id,
            "completion": completion,
            "passed": r.passed,
            "kind": r.error_kind,
            "msg": r.short_msg,
        }) + "\n")
        out.flush()

    out.close()
    n = len(tasks)
    dt = time.time() - t0
    print(f"\nFIM {variant} pass@1 = {passed}/{n} = {passed/n:.3%}")
    print(f"wallclock {dt:.0f}s, {dt/n:.2f}s/task")
    return 0


if __name__ == "__main__":
    sys.exit(main())
