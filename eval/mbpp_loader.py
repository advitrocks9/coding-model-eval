from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset


@dataclass(slots=True)
class MbppTask:
    task_id: str
    prompt: str
    entry_point: str
    test_base: str
    test_plus: str
    canonical: str


_FN_NAME = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(")


def _entry_from_code(code: str) -> str:
    m = _FN_NAME.search(code)
    if not m:
        raise ValueError(f"could not parse entry_point from {code[:80]!r}")
    return m.group(1)


def _format_prompt(description: str, first_assert: str, entry_point: str) -> str:
    # Mellum is FIM-trained, not chat-tuned. Give it the standard
    # mbpp-style docstring-with-example prompt and let it complete.
    return (
        f'"""\n{description}\n{first_assert}\n"""\n\n'
    )


def load_mbpp_plus_tasks() -> list[MbppTask]:
    ds = load_dataset("evalplus/mbppplus", split="test")
    tasks: list[MbppTask] = []
    for r in ds:
        ep = _entry_from_code(r["code"])
        first_assert = r["test_list"][0] if r["test_list"] else ""
        prompt = _format_prompt(r["prompt"], first_assert, ep)
        # build the base test from test_list
        test_base = "def check(candidate):\n"
        for assertion in r["test_list"]:
            tweaked = assertion.replace(ep, "candidate", 1)
            test_base += f"    {tweaked}\n"
        tasks.append(
            MbppTask(
                task_id=f"Mbpp/{r['task_id']}",
                prompt=prompt,
                entry_point=ep,
                test_base=test_base,
                test_plus=r["test"],
                canonical=r["code"],
            )
        )
    return tasks
