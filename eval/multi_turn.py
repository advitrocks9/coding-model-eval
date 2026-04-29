"""Multi-turn correction for FIM-style code completion models.

The setup: take any task that the model failed under HumanEval+. Re-prompt
the model with the same task plus a short comment about why the previous
attempt failed. Up to `max_extra_turns` retries. Stop early on success.

Why this matters: a model that ships in an IDE never gets a single shot.
The user runs the suggested code, it fails, the user types another prompt
or the IDE re-suggests. The relevant pass@1 is the one with at least one
round of error feedback. This module measures it.

Why short feedback: rescue review pointed out that feeding a 4B model the
full traceback can degrade output -- the model pattern-matches on traceback
noise rather than reasoning about the bug. We feed only the failing assert
line + exception kind, which is what an IDE renders inline anyway.
"""
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
    recovered_at: int | None  # turn index where plus first passes, None if never


def run_one(
    task: Task,
    generator: Generator,
    max_extra_turns: int = 2,
    retry_temperature: float = 0.6,
    use_hint: bool = True,
) -> MultiTurnResult:
    """Runs a task for up to (1 + max_extra_turns) attempts.

    Turn 0 is greedy (matches single-turn). Retries use sampling so the
    prompt change actually flips the output, with a per-task fixed seed
    so the result is reproducible. Stops early on EvalPlus pass.

    use_hint=False is the ablation: same sampling, no comment-block in the
    retry prompt. The delta between use_hint=True and False is what the
    hint actually contributes (vs. just being sampled-from-the-distribution
    luck).
    """
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
            # prefer base feedback because EvalPlus' assert helper masks the
            # actual failing input behind a generic "assert exact_match" line
            feedback = _summarise_failure(rb if not rb.passed else rp)
            prompt = build_retry_prompt(task.prompt, completion, feedback)
        # else: keep prompt = task.prompt unchanged; the only thing that
        # changes between turns is the per-turn sampling seed

    recovered_at = next((t.turn for t in turns if t.plus_passed), None)
    return MultiTurnResult(
        task_id=task.task_id,
        turns=turns,
        final_plus_passed=turns[-1].plus_passed,
        recovered_at=recovered_at,
    )


def _summarise_failure(r: ExecResult) -> str:
    """Compact error string. We avoid the full traceback on purpose."""
    if r.error_kind == "timeout":
        return "execution timed out"
    if r.error_kind == "syntax":
        return f"syntax error: {r.short_msg}"
    return r.short_msg or "additional EvalPlus tests failed"
