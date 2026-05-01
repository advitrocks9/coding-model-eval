from __future__ import annotations

from dataclasses import dataclass

from .loaders import Task
from .runner import Generator, build_retry_prompt
from .sandbox import ExecResult, execute


@dataclass(slots=True)
class TurnLog:
    turn: int
    completion: str
    base_passed: bool
    plus_passed: bool
    plus_kind: str
    plus_msg: str


@dataclass(slots=True)
class MultiTurnResult:
    task_id: str
    turns: list[TurnLog]
    final_plus_passed: bool
    recovered_at: int | None


def run_one(
    task: Task,
    generator: Generator,
    max_extra_turns: int = 2,
    retry_temperature: float = 0.6,
    use_hint: bool = True,
    hint_format: str = "current",
) -> MultiTurnResult:
    turns: list[TurnLog] = []
    prompt = task.prompt
    seed_base = abs(hash(task.task_id)) % 100_000

    for turn in range(max_extra_turns + 1):
        if turn == 0:
            completion = generator.complete(prompt)
        else:
            completion = generator.complete(prompt, temperature=retry_temperature, seed=seed_base + turn)
        full = task.prompt + completion
        rb = execute(full, task.test_base + f"\ncheck({task.entry_point})\n", timeout=10)
        rp = execute(full, task.test_plus + f"\ncheck({task.entry_point})\n", timeout=20)
        turns.append(
            TurnLog(
                turn=turn,
                completion=completion,
                base_passed=rb.passed,
                plus_passed=rp.passed,
                plus_kind=rp.error_kind,
                plus_msg=rp.short_msg,
            )
        )
        if rp.passed:
            break
        if use_hint:
            # base feedback when available; EvalPlus' wrapper masks the failing input
            feedback = _summarise_failure(rb if not rb.passed else rp)
            prompt = build_retry_prompt(task.prompt, completion, feedback, fmt=hint_format)

    recovered_at = next((t.turn for t in turns if t.plus_passed), None)
    return MultiTurnResult(
        task_id=task.task_id,
        turns=turns,
        final_plus_passed=turns[-1].plus_passed,
        recovered_at=recovered_at,
    )


def _summarise_failure(r: ExecResult) -> str:
    if r.error_kind == "timeout":
        return "execution timed out"
    if r.error_kind == "syntax":
        return f"syntax error: {r.short_msg}"
    return r.short_msg or "additional EvalPlus tests failed"
