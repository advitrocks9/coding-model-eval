from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "humaneval-infilling"

VARIANTS = {
    "single": "HumanEval-SingleLineInfilling.jsonl.gz",
    "multi": "HumanEval-MultiLineInfilling.jsonl.gz",
    "random": "HumanEval-RandomSpanInfilling.jsonl.gz",
    "light": "HumanEval-RandomSpanInfillingLight.jsonl.gz",
}


@dataclass(slots=True)
class FimTask:
    task_id: str
    entry_point: str
    prompt: str
    suffix: str
    canonical: str
    test: str


def load_fim_tasks(variant: str = "single") -> list[FimTask]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    path = DATA_DIR / VARIANTS[variant]
    with gzip.open(path, "rt") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return [
        FimTask(
            task_id=r["task_id"],
            entry_point=r["entry_point"],
            prompt=r["prompt"],
            suffix=r["suffix"],
            canonical=r["canonical_solution"],
            test=r["test"],
        )
        for r in rows
    ]
