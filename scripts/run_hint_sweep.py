# usage: python scripts/run_hint_sweep.py [MODEL_PATH] [TAG]
# runs the four retry-hint formats on the same 138 failed tasks
# (paired). compares with mcnemar offline.
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from eval.loaders import load_tasks
from eval.multi_turn import run_one
from eval.runner import Generator

DEFAULT_MODEL = "JetBrains/Mellum-4b-sft-python"
DEFAULT_TAG = "mellum_sft"
FORMATS = ("minimal", "traceback", "post")  # "current" already done in mellum_sft_multiturn.jsonl


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    single_path = Path(f"results/{tag}_singleturn.jsonl")

    tasks = load_tasks()
    by_id = {t.task_id: t for t in tasks}
    single = {row["task_id"]: row for row in (json.loads(l) for l in single_path.open())}
    failed = [by_id[tid] for tid in single if not single[tid]["plus_pass"]]
    print(f"hint-format sweep on {len(failed)} failed tasks, {len(FORMATS)} formats", flush=True)

    g = Generator(model_path)
    print("model loaded", flush=True)

    Path("results").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for fmt in FORMATS:
        out_path = Path(f"results/{tag}_hintsweep_{fmt}.jsonl")
        if out_path.exists():
            print(f"skip {fmt}: exists", flush=True)
            continue
        print(f"\n--- format = {fmt} ---", flush=True)
        out = out_path.open("w")
        recov = 0
        for t in tqdm(failed):
            res = run_one(t, g, max_extra_turns=2, hint_format=fmt)
            recov += res.final_plus_passed
            out.write(json.dumps({
                "task_id": res.task_id,
                "hint_format": fmt,
                "final_plus_passed": res.final_plus_passed,
                "recovered_at": res.recovered_at,
                "turns": [asdict(x) for x in res.turns],
            }) + "\n")
            out.flush()
        out.close()
        print(f"  {fmt} recovery: {recov}/{len(failed)} = {recov/len(failed):.1%}", flush=True)
    print(f"\nwallclock {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
