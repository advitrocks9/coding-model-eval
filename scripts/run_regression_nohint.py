# usage: python scripts/run_regression_nohint.py [MODEL_PATH] [TAG]
# Control arm for the regression test: same 26 passing tasks, sampled
# retry at the same T=0.6 used in multi-turn, but no hint. Isolates
# stochastic-retry damage from hint-induced damage.
import json
import sys
from pathlib import Path

from tqdm import tqdm

from eval.loaders import load_tasks
from eval.multi_turn import _stable_seed
from eval.runner import Generator
from eval.sandbox import execute


DEFAULT_MODEL = "/home/prannayk/models/mellum-sft-python"
DEFAULT_TAG = "mellum_sft"


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    single_path = Path(f"results/{tag}_singleturn.jsonl")
    out_path = Path(f"results/{tag}_regression_nohint.jsonl")

    tasks = load_tasks()
    by_id = {t.task_id: t for t in tasks}
    single = [json.loads(l) for l in single_path.open()]
    passing = [r for r in single if r["plus_pass"]]
    print(f"running no-hint control on {len(passing)} passing tasks")

    g = Generator(model_path)
    print("model loaded", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("w")
    regressed = 0
    for r in tqdm(passing):
        t = by_id[r["task_id"]]
        seed = _stable_seed(t.task_id) + 1
        retry = g.complete(t.prompt, temperature=0.6, seed=seed)
        full = t.prompt + retry
        rp = execute(full, t.test_plus + f"\ncheck({t.entry_point})\n", timeout=20)
        if not rp.passed:
            regressed += 1
        out.write(json.dumps({
            "task_id": t.task_id,
            "still_passes": rp.passed,
            "kind_after": rp.error_kind,
            "msg_after": rp.short_msg,
        }) + "\n")
        out.flush()
    out.close()
    n = len(passing)
    if n == 0:
        print("\nno passing tasks; nothing to control-check")
        return 0
    print(f"\nno-hint regression rate: {regressed}/{n} = {regressed/n:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
