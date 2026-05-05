# usage: python scripts/run_fim_recovery.py [MODEL_PATH] [TAG] [VARIANT]
# Multi-turn FIM recovery on the model's actual deployment shape.
# Take {tag}_he_infill_{variant}.jsonl, find tasks the model failed
# under greedy FIM completion, retry with sampled FIM completion (no
# hint, just resample). Recovery rate = how many failures pass on
# resample within 2 extra attempts.
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from eval.fim_loaders import load_fim_tasks
from eval.multi_turn import _stable_seed
from eval.runner import Generator
from eval.sandbox import execute

DEFAULT_MODEL = "JetBrains/Mellum-4b-sft-python"
DEFAULT_TAG = "mellum_sft"
DEFAULT_VARIANT = "single"
RETRY_TEMPERATURE = 0.6
MAX_EXTRA_TURNS = 2


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    variant = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_VARIANT
    single_path = Path(f"results/{tag}_he_infill_{variant}.jsonl")
    out_path = Path(f"results/{tag}_he_infill_{variant}_recovery.jsonl")

    if not single_path.exists():
        print(f"missing {single_path}; run scripts/run_humaneval_infilling.py first", flush=True)
        return 2

    tasks = load_fim_tasks(variant)
    by_id = {t.task_id: t for t in tasks}
    single = {json.loads(l)["task_id"]: json.loads(l) for l in single_path.open() if l.strip()}
    failed_ids = [tid for tid, r in single.items() if not r["passed"]]
    failed = [by_id[tid] for tid in failed_ids]
    print(f"FIM recovery on {len(failed)} failed tasks (out of {len(single)} scored)", flush=True)
    print(f"model: {model_path}\ntag: {tag}\nvariant: {variant}\nout: {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = Generator(model_path)
    print("model loaded", flush=True)

    out = out_path.open("w")
    recovered_at_turn = {1: 0, 2: 0}
    final_pass = 0
    t0 = time.time()
    for t in tqdm(failed):
        seed_base = _stable_seed(t.task_id)
        turns = []
        passed_now = False
        # turn 0 (greedy FIM) already failed -- it's why this task is in `failed`.
        # Sampled FIM retries only.
        for turn in range(1, MAX_EXTRA_TURNS + 1):
            completion = _sampled_fim(g, t, seed_base + turn)
            full = t.prompt + completion + t.suffix
            r = execute(full, t.test + f"\ncheck({t.entry_point})\n", timeout=20)
            turns.append({
                "turn": turn,
                "completion": completion,
                "passed": r.passed,
                "kind": r.error_kind,
                "msg": r.short_msg,
            })
            if r.passed:
                passed_now = True
                recovered_at_turn[turn] += 1
                break
        if passed_now:
            final_pass += 1
        out.write(json.dumps({
            "task_id": t.task_id,
            "final_passed": passed_now,
            "recovered_at": next((tn["turn"] for tn in turns if tn["passed"]), None),
            "turns": turns,
        }) + "\n")
        out.flush()
    out.close()

    n = len(failed)
    dt = time.time() - t0
    print(f"\nFIM no-hint recovery: {final_pass}/{n} = {final_pass/n:.1%}")
    print(f"  recovered at turn: {dict(sorted(recovered_at_turn.items()))}")
    print(f"wallclock {dt:.0f}s, {dt/n:.2f}s/task")
    return 0


def _sampled_fim(g: Generator, task, seed: int) -> str:
    # FIM-mode sampled: temperature 0.6, per-task seed.
    # Generator.fim_complete is greedy; reach in to sample by overriding
    # max_new_tokens and using its tokenizer/model directly.
    import torch
    wrapped = g._fim_wrap(task.prompt, task.suffix, f"{task.entry_point}.py")
    if wrapped is None:
        raise RuntimeError(f"no FIM tokens for {g.model_path}")
    ids = g.tokenizer(wrapped, return_tensors="pt", return_token_type_ids=False).to(g.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = g.model.generate(
            **ids,
            max_new_tokens=g.max_new_tokens,
            do_sample=True,
            temperature=RETRY_TEMPERATURE,
            top_p=0.95,
            pad_token_id=g.tokenizer.pad_token_id,
        )
    gen = out[0, ids["input_ids"].shape[1]:]
    text = g.tokenizer.decode(gen, skip_special_tokens=False)
    for marker in g._fim_stop_markers():
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
            break
    return text.rstrip() + "\n"


if __name__ == "__main__":
    sys.exit(main())
