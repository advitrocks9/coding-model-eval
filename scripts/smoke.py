"""Run 3 HumanEval problems through Mellum-4b-sft-python end-to-end.

Goal: confirm the model loads on the 4090, the prompt/decode loop produces
plausible Python, and the sandbox scores it correctly. Numbers don't matter
yet, just that the pipe doesn't have a hole in it.
"""
import sys

from eval.loaders import load_tasks
from eval.runner import Generator
from eval.sandbox import execute


def main() -> None:
    tasks = load_tasks()[:3]
    print(f"loading model on cuda...", flush=True)
    g = Generator("/home/prannayk/models/mellum-sft-python", device="cuda")
    print("loaded\n", flush=True)

    for t in tasks:
        completion = g.complete(t.prompt)
        full = t.prompt + completion
        rb = execute(full, t.test_base + f"\ncheck({t.entry_point})\n", timeout=10)
        rp = execute(full, t.test_plus + f"\ncheck({t.entry_point})\n", timeout=15)
        print(f"{t.task_id}: base={rb.passed}, plus={rp.passed}")
        print(f"  short err base: {rb.short_msg[:120] if not rb.passed else ''}")
        print("  ---completion (first 300 chars):")
        print("  " + completion[:300].replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    sys.exit(main())
