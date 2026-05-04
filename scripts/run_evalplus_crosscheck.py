# usage: python scripts/run_evalplus_crosscheck.py [TAG]
# Sandbox-correctness check. Take {TAG}_singleturn.jsonl, repackage as
# evalplus's expected samples format, run the official evalplus.evaluate
# CLI, and write the result JSON to results/_evalplus_crosscheck_{TAG}.json.
#
# Requires `evalplus` (install with `uv pip install -e .[crosscheck]`).
# On macOS evalplus's reliability_guard calls setrlimit, which fails
# because the soft limit is already INT64_MAX; this script monkey-patches
# the guard to a no-op for the duration of the run, which is safe for
# trusted local completions but should not be used on adversarial code.
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TAG = "mellum_sft"
TAG = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TAG


def patch_evalplus(unpatch: bool = False) -> None:
    import evalplus.eval.utils as u
    if unpatch:
        if Path(u.__file__ + ".bak").exists():
            shutil.move(u.__file__ + ".bak", u.__file__)
        return
    src = Path(u.__file__).read_text()
    Path(u.__file__ + ".bak").write_text(src)
    new_src = src.replace(
        "def reliability_guard(maximum_memory_bytes: Optional[int] = None):",
        "def reliability_guard(maximum_memory_bytes: Optional[int] = None):\n    return",
        1,
    )
    Path(u.__file__).write_text(new_src)


def main() -> int:
    src = Path(f"results/{TAG}_singleturn.jsonl")
    if not src.exists():
        print(f"missing {src}; run scripts/run_singleturn.py first", flush=True)
        return 2

    # build evalplus's samples file: {task_id, solution} where solution is
    # the full prompt + completion (the model's source as a runnable file).
    from datasets import load_dataset
    hep = {r["task_id"]: r for r in load_dataset("openai/openai_humaneval", split="test")}
    samples = Path(f"/tmp/{TAG}_evalplus_samples.jsonl")
    rows = [json.loads(l) for l in src.open()]
    with samples.open("w") as f:
        for r in rows:
            full = hep[r["task_id"]]["prompt"] + r["completion"]
            f.write(json.dumps({"task_id": r["task_id"], "solution": full}) + "\n")
    print(f"wrote {samples} ({len(rows)} samples)", flush=True)

    # remove any cached eval result so evalplus re-runs
    cache = samples.with_name(samples.stem + "_eval_results.json")
    cache.unlink(missing_ok=True)

    # patch evalplus, run it, copy the result, unpatch
    if sys.platform == "darwin":
        patch_evalplus(unpatch=False)
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "evalplus.evaluate",
                "--dataset", "humaneval",
                "--samples", str(samples),
                "--parallel", "1",
                "--i_just_wanna_run",
            ],
            check=False,
        )
    finally:
        if sys.platform == "darwin":
            patch_evalplus(unpatch=True)

    if not cache.exists():
        print(f"evalplus didn't write {cache}; rc={proc.returncode}", flush=True)
        return proc.returncode or 1

    out = Path(f"results/_evalplus_crosscheck_{TAG}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cache, out)
    print(f"wrote {out}", flush=True)

    # quick agreement summary
    d = json.loads(out.read_text())
    me = {r["task_id"]: (r["base_pass"], r["plus_pass"]) for r in rows}
    them = {tid: (v[0]["base_status"] == "pass", v[0]["plus_status"] == "pass")
            for tid, v in d["eval"].items()}
    base_agree = sum(1 for tid in me if me[tid][0] == them.get(tid, (None,))[0])
    plus_agree = sum(1 for tid in me if me[tid][1] == them.get(tid, (None, None))[1])
    me_base = sum(b for b, _ in me.values()); me_plus = sum(p for _, p in me.values())
    th_base = sum(b for b, _ in them.values()); th_plus = sum(p for _, p in them.values())
    n = len(me)
    print(f"  base: me={me_base}/{n} ({me_base/n:.1%}), evalplus={th_base}/{n} ({th_base/n:.1%}), per-task agree={base_agree}/{n}")
    print(f"  plus: me={me_plus}/{n} ({me_plus/n:.1%}), evalplus={th_plus}/{n} ({th_plus/n:.1%}), per-task agree={plus_agree}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
