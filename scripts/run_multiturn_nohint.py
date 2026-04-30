"""Ablation: same multi-turn loop, but no comment hint on retries.

The point. My multi-turn run improved EvalPlus pass@1 by 1.2 pp on Mellum-SFT.
That number contains two effects: (a) the model sees a hint and tries
something different; (b) the retry uses temperature 0.6 sampling instead
of greedy, so any retry would generate something different anyway. This
script isolates (b). If recovery rate is similar without the hint, the
hint isn't doing real work.

usage: python scripts/run_multiturn_nohint.py [MODEL_PATH] [TAG]
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from eval.loaders import load_tasks
from eval.multi_turn import run_one
from eval.runner import Generator

DEFAULT_MODEL = "/home/prannayk/models/mellum-sft-python"
DEFAULT_TAG = "mellum_sft"


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    single_path = Path(f"results/{tag}_singleturn.jsonl")
    out_path = Path(f"results/{tag}_multiturn_nohint.jsonl")

    tasks = load_tasks()
    single = {row["task_id"]: row for row in (json.loads(l) for l in single_path.open())}
    g = Generator(model_path)
    print("model loaded", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("w")

    n_total = len(tasks)
    n_already = sum(1 for r in single.values() if r["plus_pass"])
    n_retry = n_total - n_already
    print(f"{n_already}/{n_total} already pass; retrying {n_retry} with no hint", flush=True)

    final_pass = n_already
    recovered = 0
    t0 = time.time()
    for t in tqdm(tasks):
        if single[t.task_id]["plus_pass"]:
            out.write(json.dumps({
                "task_id": t.task_id,
                "skipped_reason": "already_passed",
                "final_plus_passed": True,
                "recovered_at": 0,
                "turns": [{
                    "turn": 0,
                    "completion": single[t.task_id]["completion"],
                    "base_passed": single[t.task_id]["base_pass"],
                    "plus_passed": True,
                    "plus_kind": "ok",
                    "plus_msg": "",
                }],
            }) + "\n")
            continue
        res = run_one(t, g, max_extra_turns=2, use_hint=False)
        final_pass += res.final_plus_passed
        recovered += res.final_plus_passed
        out.write(json.dumps({
            "task_id": res.task_id,
            "final_plus_passed": res.final_plus_passed,
            "recovered_at": res.recovered_at,
            "turns": [asdict(t) for t in res.turns],
        }) + "\n")
        out.flush()

    out.close()
    dt = time.time() - t0
    print(f"\nplus pass@1 (no-hint multi-turn) = {final_pass}/{n_total} = {final_pass/n_total:.3%}")
    print(f"recovered without hint: {recovered} ({recovered/n_retry:.1%} of {n_retry})")
    print(f"wallclock {dt:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
