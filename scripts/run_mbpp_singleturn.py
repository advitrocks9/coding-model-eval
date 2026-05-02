# usage: python scripts/run_mbpp_singleturn.py [MODEL_PATH] [TAG]
# MBPP+ single-turn baseline. Prompt is mbpp-style: docstring with the
# first assertion as a usage example, model writes the function.
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from eval.mbpp_loader import load_mbpp_plus_tasks
from eval.runner import Generator
from eval.sandbox import execute

DEFAULT_MODEL = "/home/prannayk/models/mellum-sft-python"
DEFAULT_TAG = "mellum_sft"


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    out_path = Path(f"results/{tag}_mbpp_singleturn.jsonl")

    tasks = load_mbpp_plus_tasks()
    print(f"loaded {len(tasks)} MBPP+ tasks", flush=True)
    print(f"model: {model_path}\ntag: {tag}\nout: {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = Generator(model_path)
    print("model loaded", flush=True)

    out = out_path.open("w")
    base_pass = plus_pass = 0
    t0 = time.time()
    for t in tqdm(tasks):
        completion = g.complete(t.prompt)
        full = t.prompt + completion
        rb = execute(full, t.test_base + f"\ncheck({t.entry_point})\n", timeout=10)
        rp = execute(full, t.test_plus + f"\ncheck({t.entry_point})\n", timeout=20)
        base_pass += rb.passed
        plus_pass += rp.passed
        out.write(json.dumps({
            "task_id": t.task_id,
            "completion": completion,
            "base_pass": rb.passed,
            "base_kind": rb.error_kind,
            "base_msg": rb.short_msg,
            "plus_pass": rp.passed,
            "plus_kind": rp.error_kind,
            "plus_msg": rp.short_msg,
        }) + "\n")
        out.flush()

    out.close()
    n = len(tasks)
    dt = time.time() - t0
    print(f"\nbase pass@1 = {base_pass}/{n} = {base_pass/n:.3%}")
    print(f"plus pass@1 = {plus_pass}/{n} = {plus_pass/n:.3%}")
    print(f"wallclock {dt:.0f}s, {dt/n:.2f}s/task")
    return 0


if __name__ == "__main__":
    sys.exit(main())
