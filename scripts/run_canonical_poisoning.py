# usage: python scripts/run_canonical_poisoning.py [MODEL_PATH] [TAG]
# Shared-task cross-family poisoning assay: every model gets the SAME
# 164 tasks with the SAME canonical solution as the "previous attempt",
# then the current retry hint, then regenerate greedy. Removes the
# exposure-set confound where DS-base regression is over 41 tasks vs
# Mellum-SFT's 26 etc. The metric is "given a known-correct solution
# and the current hint, how often does the regeneration still pass
# the plus tests."
import json
import sys
from pathlib import Path

from tqdm import tqdm

from eval.loaders import load_tasks
from eval.runner import Generator, build_retry_prompt
from eval.sandbox import execute


DEFAULT_MODEL = "JetBrains/Mellum-4b-sft-python"
DEFAULT_TAG = "mellum_sft"
FAKE_HINT = "expected output X, got Y (synthetic hint for the canonical poisoning assay)"


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    out_path = Path(f"results/{tag}_canonical_poisoning.jsonl")

    tasks = load_tasks()
    print(f"running canonical-solution poisoning on {len(tasks)} tasks")

    g = Generator(model_path)
    print("model loaded", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("w")
    broken = 0
    for t in tqdm(tasks):
        prompt = build_retry_prompt(t.prompt, t.canonical, FAKE_HINT)
        retry = g.complete(prompt)
        full = t.prompt + retry
        rp = execute(full, t.test_plus + f"\ncheck({t.entry_point})\n", timeout=20)
        if not rp.passed:
            broken += 1
        out.write(json.dumps({
            "task_id": t.task_id,
            "still_passes": rp.passed,
            "kind_after": rp.error_kind,
            "msg_after": rp.short_msg,
        }) + "\n")
        out.flush()
    out.close()
    n = len(tasks)
    print(f"\ncanonical-poisoning breakage: {broken}/{n} = {broken/n:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
