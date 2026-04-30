# coding-model-eval

A small set of experiments on whether HumanEval pass@1 actually measures
what people quote it for. The thing I keep coming back to is that the
benchmark a model is graded on quietly determines what it learns to
optimise, so before believing any pass@1 number I want to know how
sensitive it is to test rigour, retry shape, and whether the model is
being run in the format it was trained on.

I picked Mellum-4b because JetBrains released three variants of it
(base, SFT-on-Python, DPO-on-Python). Same architecture, same
pre-training, three post-training stages. That's the cleanest
in-the-wild lever I could pull. Then I added EvalPlus's augmented
test suite, a multi-turn retry loop, a no-hint ablation, and a
regression test that flips the framing on multi-turn.

## Headline

| tag | base | plus | + retry (hint) | + retry (no hint) | regression |
|---|---:|---:|---:|---:|---:|
| Mellum-4b-base (completion mode) | 24.4% | 21.3% | – | – | – |
| Mellum-4b-base (FIM tokens) | 23.8% | 20.7% | – | – | – |
| Mellum-4b-sft-python | 18.3% | 15.9% | 17.1% | **22.6%** | 8/26 (31%) |
| Mellum-4b-dpo-python | 11.0% | 9.1% | 9.8% | – | 4/15 (27%) |

Five things worth noticing.

1. **Each post-training stage hurts HumanEval pass@1.** Base is best
   (24.4%), SFT is worse (18.3%), DPO is worst (11.0%). JetBrains
   evaluates Mellum on FIM benchmarks (RepoBench, SAFIM, HumanEval
   Infilling) and presumably moves those numbers in the opposite
   direction. So "Mellum on HumanEval" is a measurement that gets
   *worse* across the training pipeline that ships the model. Which
   says more about the benchmark than about the model.
2. **The EvalPlus penalty scales with model strength.** EvalPlus
   reports 13–30 pp drops on stronger models. Mellum-base has a 3 pp
   penalty, SFT has 2.4, DPO has 1.8. Below a certain capability
   level there's just less inflation to remove. The EvalPlus penalty
   isn't a fixed measure of test rigour; it's a measure of test rigour
   weighted by how much room the model has to game weak tests.
3. **Multi-turn retry with a hint barely recovers anything.** On
   Mellum-SFT, recovery rate is 2/138 = 1.4%. I went in expecting
   double-digit recovery and a story about "self-correction is free."
   That isn't what the data shows.
4. **The retry hint actively damages correct solutions.** The
   regression test takes the 26 tasks the model already passes,
   fabricates a "previous attempt was wrong, try again" hint, and
   reruns. **8 of 26 = 30.8% break.** Greedy decoding, deterministic,
   only the hint changed. So the +1.2 pp the multi-turn pipeline
   reports is a selection artefact: it only looks like a gain because
   the loop has access to ground-truth tests and skips the
   already-correct cases. A naive "always retry" pipeline applied to
   all 164 tasks would post lower pass@1 than single-turn.
5. **The hint isn't doing useful work; it's the noise.** Same retry
   loop, same temperature 0.6 sampling, no comment hint: recovery
   jumps to 11/138 = 8.0%, and pass@1 jumps to 22.6%. **5.7× more
   problems recover when you remove the hint.** The right multi-turn
   policy on Mellum-SFT is "resample, don't explain"; the explanation
   is the part that's hurting.

The plot, briefly:

![recovery vs regression](assets/recovery_vs_regression.png)

Both with-hint Mellum points sit below the y=x diagonal. The no-hint
ablation point sits in the upper-left because it has no hint to break
correct solutions. That's the comparison I'd put on a slide.

## Why does the hint hurt

I don't think this is a "small models can't reason" story. The retries
*do* recover problems when the prompt is left alone and the model is
just sampled differently. So the model knows things; it isn't using
them when the comment block is in the prompt.

My read: Mellum is a code-completion model, trained on raw Python files.
The retry hint format is a comment block above the function that
mentions the previous attempt and the failing assertion. The model
treats that as code it should honor, not as instructions to ignore. So
on retry it tries to literally produce the commented-out attempt or to
patch around the literal assertion, instead of writing a clean function.
Inspecting the per-turn outputs shows exactly this: lots of
commented-out previous attempts copied forward, and `pass`-only bodies.
A chat-tuned model would presumably handle the same hint differently;
this is part of why I'd test that next.

## Multi-turn isn't free, even when it looks free

The way pass@k is usually reported for retry-style evaluation is "pass@1
single-turn vs pass@1 with up to k retries." That number folds two
things together: how often retries fix mistakes, and how the retry
mechanism behaves when the previous attempt was already correct. In a
benchmark eval that always knows ground truth and so only retries on
observed failures, the second thing never gets exercised. In production
the second thing is most of the population: most of the time the model
is right and the user retries because they want a different style or
because they're impatient.

The minimum reportable number for any "multi-turn helps" claim is the
pair (recovery rate, regression rate), where regression rate is
measured on a held-out slice the model already passes. Reporting one
without the other smuggles in the assumption that a deployment has
ground-truth tests for every input, and it doesn't.

## What's in here

```
eval/sandbox.py     subprocess + 10s timeout, returns pass/fail and a
                    short feedback string. -I flag on the interpreter
                    because /tmp/inspect.py once ate two hours.
eval/loaders.py     joins openai/openai_humaneval and evalplus/humanevalplus
                    on task_id so one completion is scored under both
                    test suites
eval/runner.py      AutoModel wrapper with a normal completion mode and
                    a FIM-mode completion using <fim_prefix> /
                    <fim_suffix> / <fim_middle>
eval/multi_turn.py  retry loop with a use_hint flag for the ablation
eval/report.py      cross-table for every model tag found in results/
scripts/            one runner per experiment, all take MODEL_PATH and
                    TAG positional args
results/            JSONL per (tag, experiment), one row per task
```

About 600 lines of Python.

## Things that didn't work, and one thing I want to flag

- First multi-turn run produced byte-identical output on every retry.
  Greedy decoding plus a small comment-prefixed prompt rarely flips
  the argmax, so the "retry" was a no-op. Switched to temperature 0.6
  with a per-task seed for retries only. Turn 0 stays greedy so the
  single-turn number matches what a leaderboard would post.
- Halfway through, every plus test started failing with
  `AttributeError: module 'inspect' has no attribute 'cleandoc'`.
  Cause: I'd dropped a `/tmp/inspect.py` while debugging. Python adds
  the script's directory to `sys.path[0]` when running a file, so my
  one-line shadow file beat stdlib inspect, then numpy fails to
  import inside the EvalPlus test fixture. Sandbox now passes `-I` to
  the interpreter. Worth flagging if you write a sandbox that puts
  test scripts in a shared tmpdir.
- I expected Mellum-SFT to outperform Mellum-base on HumanEval. The
  opposite ordering (base > SFT > DPO) is the writeup.
- I expected the retry hint to help recovery. The opposite is true.

## What I'd do next if I had another week

- **Run the same battery on a chat-tuned model.** Qwen2.5-Coder-Instruct
  is the obvious comparison. The hypothesis: the hint helps a chat-tuned
  model and hurts a code-completion model, because they read comments
  differently. If that holds, a leaderboard that reports a single
  multi-turn pass@1 number without specifying hint format is just
  conflating two different evaluations.
- **FIM-aware test mutation.** EvalPlus's mutation strategy doesn't
  transfer to FIM benchmarks. The right shape is to mutate the
  *surrounding file context* (rename variables, reorder imports, add
  an unrelated function above), AST-level, and re-evaluate. That gives
  a robustness number in the deployment-relevant direction.
- **Wider sampling instead of more retries.** No-hint recovery already
  beats hint recovery at the same retry budget. Pass@k with diverse
  samples might beat both, and is cheaper than reasoning-style multi-turn.

## Reproducing

Runs were on a 4090 (24GB). Wallclocks for the full Mellum-4b family:

```bash
uv sync
PYTHONPATH=. uv run python scripts/run_singleturn.py /home/<u>/models/mellum-sft-python mellum_sft         # ~13 min
PYTHONPATH=. uv run python scripts/run_multiturn.py  /home/<u>/models/mellum-sft-python mellum_sft         # ~45 min
PYTHONPATH=. uv run python scripts/run_regression.py /home/<u>/models/mellum-sft-python mellum_sft         # ~5 min
PYTHONPATH=. uv run python scripts/run_multiturn_nohint.py /home/<u>/models/mellum-sft-python mellum_sft   # ~30 min
PYTHONPATH=. uv run python scripts/run_singleturn.py /home/<u>/models/mellum-base mellum_base              # ~18 min
PYTHONPATH=. uv run python scripts/run_fim.py        /home/<u>/models/mellum-base mellum_base              # ~5 min
PYTHONPATH=. uv run python -m eval.report
PYTHONPATH=. uv run python scripts/make_plot.py
```

The torch index in `pyproject.toml` is pinned to cu128 so the lockfile
resolves on a fresh CUDA box. To run on Apple silicon, drop the
`[tool.uv.sources]` block and let uv resolve a CPU build.

## Things this is not

This is a small probe, not a benchmark. One model family, one benchmark
family, one decoding configuration. The conclusions worth defending
from this much data are "the way pass@1 with retries is reported needs
more components than it currently has" and "Mellum's HumanEval scores
under-represent the model in two specific ways." Stronger versions of
either claim would require running this on at least one chat-tuned
family for comparison.

Underlying paper that started this:
[Liu et al., NeurIPS 2023, "Is Your Code Generated by ChatGPT Really
Correct?"](https://arxiv.org/abs/2305.01210). Mellum models from
JetBrains are on [HuggingFace](https://huggingface.co/JetBrains).
