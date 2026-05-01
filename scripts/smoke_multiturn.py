import sys

from eval.loaders import load_tasks
from eval.multi_turn import run_one
from eval.runner import Generator


def main() -> int:
    tasks = load_tasks()
    by_id = {t.task_id: t for t in tasks}
    sample = [by_id["HumanEval/0"], by_id["HumanEval/3"], by_id["HumanEval/5"]]

    g = Generator("/home/prannayk/models/mellum-sft-python")
    print("model loaded")
    for t in sample:
        r = run_one(t, g, max_extra_turns=2)
        print(f"\n=== {t.task_id} final={r.final_plus_passed} recovered_at={r.recovered_at} ===")
        for tl in r.turns:
            print(f"  turn {tl.turn} base={tl.base_passed} plus={tl.plus_passed}")
            print("    completion:", tl.completion.strip()[:100])
    return 0


if __name__ == "__main__":
    sys.exit(main())
