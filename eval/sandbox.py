"""Run untrusted Python in a subprocess with a wallclock timeout.

The point: HumanEval and EvalPlus tests are arbitrary Python with assert
statements. Run them in a child process so a hang or crash can't take down
the runner. No Docker, no namespacing, just timeout + tmpfile + a clean exit.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExecResult:
    passed: bool
    error_kind: str  # "ok" | "timeout" | "assertion" | "exception" | "syntax"
    short_msg: str   # one-line summary used as multi-turn feedback


_ASSERT_RE = re.compile(r"^\s*assert\b", re.MULTILINE)


def execute(code: str, test: str, timeout: float = 10.0) -> ExecResult:
    """Run `code` then `test` in a subprocess. test is a snippet that
    asserts on `code`'s exported names. Returns pass/fail + a feedback hint
    short enough to feed back into a chat model without flooding it.
    """
    script = code + "\n\n" + test
    tmp = Path(tempfile.mkstemp(suffix=".py")[1])
    try:
        tmp.write_text(script)
        # -I (isolated) prevents Python from prepending the script's directory
        # to sys.path. Without this, a stray /tmp/inspect.py or /tmp/json.py
        # silently shadows the stdlib for the whole run. Asked me 30 minutes.
        proc = subprocess.run(
            [sys.executable, "-I", str(tmp)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(False, "timeout", f"timed out after {timeout:.0f}s")
    finally:
        tmp.unlink(missing_ok=True)

    if proc.returncode == 0:
        return ExecResult(True, "ok", "")

    err = proc.stderr.strip()
    last = err.splitlines()[-1] if err else "unknown error"
    if "AssertionError" in err:
        # find the failing assert line if we can
        match = re.search(r'assert .+', err)
        line = match.group(0) if match else last
        return ExecResult(False, "assertion", line[:200])
    if "SyntaxError" in err or "IndentationError" in err:
        return ExecResult(False, "syntax", last[:200])
    return ExecResult(False, "exception", last[:200])
