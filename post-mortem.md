# Post-mortem: an empty file in `/tmp` poisoned half a day of runs

## What happened

Mid-way through the multi-turn experiment on Mellum-SFT, every retry on every
task started failing with the same error:

```
AttributeError: module 'inspect' has no attribute 'cleandoc'
```

The first multi-turn run (164 tasks, ~45 minutes on the 4090) reported zero
recoveries: every retry failed at module-import time before the test even
ran. Initially I read this as "the model produces uncompilable code on
retry", which is wrong, but plausible enough that I almost wrote it up
that way.

## What I checked, in order

1. **Is `inspect.cleandoc` actually missing?** No, it's been in stdlib
   since Python 3.2. `python -c "import inspect; print(inspect.cleandoc)"`
   in the venv prints `<function cleandoc at 0x...>`.
2. **Does `inspect.cleandoc` work in a subprocess?** Same `python -c`, but
   from inside `subprocess.run`. Still works. So the parent's interpreter
   is fine.
3. **What's actually triggering it?** Not my code directly. The traceback
   said the error came from `numpy/_core/overrides.py:161`, inside numpy's
   import chain. Numpy calls `inspect.cleandoc(dispatcher.__doc__)` while
   building its array dispatcher. So the `import numpy` that the EvalPlus
   test fixture does is what's failing, and `inspect` is the proximate
   cause.

That last hop was the lightbulb. If `inspect.cleandoc` exists in stdlib but
isn't visible to `import inspect` from inside the subprocess, then the
`inspect` being imported isn't stdlib's.

## The actual cause

Earlier in the session I'd dumped a debugging script at `/tmp/inspect.py`
to run a one-off check on the JSONL output. Five lines, called
`json.load`, no relation to `inspect`. The file just happened to be named
`inspect.py`.

Python's interpreter inserts `sys.path[0] = <directory of the script
being run>` at startup. My sandbox writes the test code to a fresh tmpfile
via `tempfile.mkstemp(suffix=".py")`, which by default goes into `/tmp`.
So when I run `subprocess.run([sys.executable, str(tmp)])`, the subprocess
starts with `sys.path[0] = "/tmp"`. Then numpy's `from . import multiarray`
does an import that eventually reaches a bare `import inspect`. Python
walks `sys.path`, finds `/tmp/inspect.py`, and imports my five-line file
as the `inspect` module. Numpy's call to `inspect.cleandoc` then raises
`AttributeError`, because of course my one-off script doesn't define
`cleandoc`.

## The fix

```python
proc = subprocess.run([sys.executable, "-I", str(tmp)], ...)
```

`python -I` is "isolated mode" (added in Python 3.4 via the work that
also produced `python -E` and `-s`). It does several things; the one I
wanted here is that it does **not** prepend the script's directory to
`sys.path`. With `-I`, the subprocess can't see `/tmp/inspect.py`, or
any other shadowing file in the script's directory.

I could also have written the test scripts to a private subdirectory
created per-run, or run them through `python -m` and a fixed import,
or used `subprocess.run(..., cwd=Path(tempfile.mkdtemp()))`. `-I` is the
smallest correct change. The comment in `eval/sandbox.py` flags it so
future me doesn't undo it.

## Why this matters beyond my project

Any code-eval pipeline that writes temporary scripts and runs them with
the host interpreter has this bug latent. EvalPlus's evaluator, the
`human-eval` package, and several public eval harnesses all dump test
files into a temp dir and exec them. If your tmpdir happens to contain
a Python file matching the name of any module imported by your tests,
or by any module those tests import, all your tests fail with a
confusing error. My sandbox is small enough that I caught it in 30
minutes; in a larger pipeline with timeouts swallowing stderr, the
silent corruption could go unnoticed for a long time.

The general lesson: subprocess inherits more environment from its
parent's *current working directory* than is obvious. `cwd`, `sys.path[0]`,
and any `.pth` files in the script's directory all flow through. If the
sandbox writes scripts to a path that anyone else can also write to, you
have a confused-deputy bug waiting to happen.

For a production eval pipeline I'd probably do all three: `-I`, write to a
per-run tempdir, and assert at startup that the tempdir has no `.py`
files in it.

## What I'd do differently next time

- I would have pulled the actual stack trace out of stderr earlier. I was
  staring at "AttributeError: module 'inspect' has no attribute 'cleandoc'"
  in the JSONL `short_msg` field for half an hour before I went looking
  for the full traceback. The sandbox truncates stderr to one line for
  the multi-turn hint feeder, which is the right call for the model but
  the wrong call for me. A `--verbose` flag that keeps full stderr in the
  JSONL would have let me see the numpy line on row 1.
- I would have noticed the partial truth that "every task fails with the
  same error" was already telling me the bug wasn't in any specific task.
  Same error across 164 different prompts means the bug is in the
  fixture, not the model. I should have looked at the fixture
  (EvalPlus test field) before looking at the model output.

## Takeaway for the project

Single line change to the sandbox, half a paragraph of explanation in a
code comment. The lesson worth remembering is about subprocess sandboxes,
not the fix itself. I'm leaving this writeup in the repo because
future-me will absolutely write `/tmp/foo.py` again and forget that this
is a thing.
