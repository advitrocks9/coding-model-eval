# usage: python scripts/run_fim.py [MODEL_PATH] [TAG]
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from eval.loaders import load_tasks
from eval.runner import Generator
from eval.sandbox import execute

DEFAULT_MODEL = "/home/prannayk/models/mellum-base"
DEFAULT_TAG = "mellum_base_fim"


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    out_path = Path(f"results/{tag}_fim.jsonl")

    tasks = load_tasks()
    print(f"loaded {len(tasks)} tasks", flush=True)
    print(f"model: {model_path}\ntag: {tag}\nout: {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = Generator(model_path)
    print("model loaded", flush=True)

    out = out_path.open("w")
    base_pass = plus_pass = 0
    t0 = time.time()
    for t in tqdm(tasks):
        completion = g.fim_complete(t.prompt, suffix="\n", filename=f"{t.entry_point}.py")
        full_code = t.prompt + completion
        rb = execute(full_code, t.test_base + f"\ncheck({t.entry_point})\n", timeout=10)
        rp = execute(full_code, t.test_plus + f"\ncheck({t.entry_point})\n", timeout=20)
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
    dt = time.time() - t0
    n = len(tasks)
    print(f"\nFIM mode")
    print(f"base pass@1 = {base_pass}/{n} = {base_pass/n:.3%}")
    print(f"plus pass@1 = {plus_pass}/{n} = {plus_pass/n:.3%}")
    print(f"wallclock {dt:.0f}s, {dt/n:.1f}s/task")
    return 0


if __name__ == "__main__":
    sys.exit(main())
