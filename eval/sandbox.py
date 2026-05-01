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
    error_kind: str
    short_msg: str


def execute(code: str, test: str, timeout: float = 10.0) -> ExecResult:
    script = code + "\n\n" + test
    tmp = Path(tempfile.mkstemp(suffix=".py")[1])
    try:
        tmp.write_text(script)
        # -I prevents the subprocess from picking up /tmp/inspect.py etc.
        # see post-mortem.md.
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
        match = re.search(r"assert .+", err)
        line = match.group(0) if match else last
        return ExecResult(False, "assertion", line[:200])
    if "SyntaxError" in err or "IndentationError" in err:
        return ExecResult(False, "syntax", last[:200])
    return ExecResult(False, "exception", last[:200])
