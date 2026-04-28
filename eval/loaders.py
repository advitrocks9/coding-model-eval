"""Load HumanEval and HumanEval+ tasks into a common shape.

EvalPlus retains the original HumanEval prompt and entry_point, but rewrites
the test suite. So a single completion can be scored under both. That's the
point of the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset


@dataclass(slots=True)
class Task:
    task_id: str           # e.g. "HumanEval/0"
    prompt: str            # the function signature + docstring
    entry_point: str       # function name
    test_base: str         # original HumanEval `test` field
    test_plus: str         # EvalPlus' augmented `test` field
    canonical: str         # reference solution, used for sanity checks
    base_input_count: int  # how many inputs in the base test set
    plus_extra_count: int  # how many extra inputs EvalPlus adds


def load_tasks() -> list[Task]:
    """Join openai/openai_humaneval and evalplus/humanevalplus on task_id."""
    base = {row["task_id"]: row for row in load_dataset("openai/openai_humaneval", split="test")}
    plus = {row["task_id"]: row for row in load_dataset("evalplus/humanevalplus", split="test")}

    common = sorted(set(base) & set(plus), key=_he_sort_key)
    tasks: list[Task] = []
    for tid in common:
        b, p = base[tid], plus[tid]
        # heuristic counts so the writeup can quote how many extra tests EvalPlus added
        base_n = b["test"].count("assert ")
        plus_n = p["test"].count("assertion(") - base_n
        tasks.append(
            Task(
                task_id=tid,
                prompt=b["prompt"],
                entry_point=b["entry_point"],
                test_base=b["test"],
                test_plus=p["test"],
                canonical=b["canonical_solution"],
                base_input_count=base_n,
                plus_extra_count=max(plus_n, 0),
            )
        )
    return tasks


def _he_sort_key(tid: str) -> int:
    return int(tid.split("/")[1])


def select(tasks: list[Task], task_ids: Iterable[str]) -> list[Task]:
    keep = set(task_ids)
    return [t for t in tasks if t.task_id in keep]
