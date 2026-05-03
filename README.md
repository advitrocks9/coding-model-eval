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
test suite, a multi-turn retry loop, a no-hint ablation, a regression
test that flips the framing on multi-turn, OpenAI's HumanEval-Infilling
to score Mellum on what it's actually trained for, and a cross-family
check on DeepSeek-Coder-1.3B (base + instruct).

## Headline

Pass@1 numbers; [low, high] are Wilson 95% CIs. n=164 for single-turn,
138 for retries (the tasks the model fails on the first try). Regression
n equals the number of tasks the model passes single-turn: 26 for
Mellum-SFT, 15 for DPO, 41 for DeepSeek-base, 90 for DeepSeek-instruct.

| tag | HumanEval | HumanEval+ | + retry (hint) | + retry (no hint) | regression |
|---|---:|---:|---:|---:|---:|
| Mellum-4b-base (completion) | 24.4% [18.5, 31.5] | 21.3% [15.8, 28.2] | – | – | – |
| Mellum-4b-base (FIM tokens) | 23.8% [17.9, 30.8] | 20.7% [15.2, 27.6] | – | – | – |
| Mellum-4b-sft-python | 18.3% [13.1, 24.9] | 15.9% [11.1, 22.2] | 17.1% [12.1, 23.6] | **22.6% [16.7, 29.7]** | 8/26 = 30.8% [16.5, 50.0] |
| Mellum-4b-dpo-python | 11.0% [7.1, 16.7] | 9.1% [5.6, 14.5] | 9.8% [6.1, 15.3] | – | 4/15 = 26.7% [10.9, 52.0] |
| DeepSeek-Coder-1.3B-base | 29.3% [22.8, 36.6] | 25.0% [19.0, 32.1] | 28.0% [21.7, 35.4] | 37.2% [30.2, 44.7] | 1/41 = 2.4% [0.4, 12.6] |
| DeepSeek-Coder-1.3B-instruct | 57.3% [49.7, 64.6] | 54.9% [47.2, 62.3] | 63.4% [55.8, 70.4] | 67.1% [59.6, 73.8] | 4/90 = 4.4% [1.7, 10.9] |

### Mellum's FIM benchmark — post-training direction reverses

JetBrains evaluates Mellum on FIM benchmarks (RepoBench, SAFIM,
HumanEval-Infilling), not HumanEval. The HumanEval table above is
the wrong axis for this family. So I ran the OpenAI
HumanEval-Infilling single-line benchmark (1033 tasks) and MBPP+
(378 tasks, free-form description prompt) on the full 5-model set:

| | HumanEval+ | HumanEval-Infilling | MBPP+ |
|---|---:|---:|---:|
| Mellum-4b-base | 21.3% | 76.9% [74.2, 79.3] | – |
| Mellum-4b-sft-python | 15.9% | 80.8% [78.3, 83.1] | **1.3% [0.6, 3.1]** |
| Mellum-4b-dpo-python | 9.1% | 81.9% [79.4, 84.1] | – |
| DS-Coder-1.3B-base | 25.0% | 13.0% [11.0, 15.2] | 22.0% [18.1, 26.4] |
| DS-Coder-1.3B-instruct | 54.9% | 3.7% [2.7, 5.0] | 52.6% [47.6, 57.6] |

Three things this 3-benchmark grid pins down. (1) Same Mellum-DPO,
9.1% on HumanEval+ vs 81.9% on HumanEval-Infilling, **9× difference**
on the *same model* — the benchmark choice is doing more work than
the model choice. (2) The post-training direction reversal is
strongest at the base→SFT step. Paired McNemar on the 1033 FIM
tasks: base→SFT p=0.001 \*\*, base→DPO p<0.001 \*\*, but **SFT→DPO
p=0.25 (NS)** — the additional FIM gain from DPO over SFT is
suggestive, not significant at this n. The HumanEval direction
(each stage hurts) is consistent across all comparisons. So the
defensible version of "opposite signs" is base vs post-trained,
not stage-by-stage. (3) Mellum-SFT scores **1.3% on MBPP+** —
basically zero — because MBPP+'s prompt format (docstring at the
top of an empty file) is out of distribution for a FIM-on-Python
model. Drop the same task into a partially-completed file with
`<fim_prefix>`/`<fim_suffix>` tokens and the model scores 80%+.
The format is doing all the work. *Caveat:* my MBPP+ prompt is a
minimal docstring scaffold; a paper-matched prompt could give a
higher number, so 1.3% is an upper-bound argument for "format-OOD"
not a tight measurement. (DS-Coder, trained on more diverse
data, handles all three benchmark formats.)

The Mellum-base FIM number, 76.9%, is 3.5 pp under the JetBrains
paper's published 80.45%. I haven't done a paper-matched prompt
ablation; the within-pipeline orderings are still internally
consistent (the Mellum FIM scores all fall in a tight band) but a
3.5 pp absolute miss against the only external anchor is real and
worth noting.


CIs matter for recovery: with-hint recovery on Mellum-SFT is 1.4%
[0.4, 5.1]; no-hint recovery is 8.0% [4.5, 13.7]. The intervals don't
overlap, so the 5.7× effect is real, not sampling noise on n=138.

### Compute-matched comparison

The interesting question isn't "does multi-turn help" but "what's the
best policy at a fixed compute budget." On Mellum-SFT, with at most 3
generations per task:

| strategy | pass@1 | what it does |
|---|---:|---|
| single-turn greedy | 15.9% | one greedy completion |
| pass@3 sampled at T=0.6 | 18.8% | three independent samples, no retry mechanism |
| multi-turn with hint (3 attempts) | 17.1% | greedy then up to 2 sampled retries with comment-block hint |
| **multi-turn no hint (3 attempts)** | **22.6%** | greedy then up to 2 sampled retries, no hint, just resample |

So sampling beats greedy (+2.9 pp), but greedy-anchored sampling beats
free sampling (+3.8 pp), and adding the hint to the retries claws back
5.5 pp of that gain. The greedy turn 0 is doing real work that pure
pass@k loses. The retry mechanism does real work too, as long as you
don't poison it with the wrong hint format. (At pass@4 the budget mismatch
goes the other way: pass@4 is 20.1%, multi-turn-no-hint is 22.6% on a
3-generation budget, so 4 free samples don't quite catch the 3-attempt
greedy-anchored policy.)

Five things worth noticing.

1. **Each post-training stage hurts HumanEval pass@1.** Base is best
   (24.4%), SFT is worse (18.3%), DPO is worst (11.0%). JetBrains
   evaluates Mellum on FIM benchmarks (RepoBench, SAFIM, HumanEval
   Infilling) and presumably moves those numbers the other way.
   "Mellum on HumanEval" is a measurement that gets *worse* across
   the training pipeline that ships the model.
2. **The EvalPlus penalty scales with model strength.** EvalPlus
   reports 13–30 pp drops on stronger models; Mellum-base has 3 pp,
   SFT 2.4, DPO 1.8. Below a certain capability level there's less
   inflation to remove. The EvalPlus penalty isn't a fixed property
   of the test suite; it's test rigour weighted by how much room the
   model has to game weak tests.
3. **The retry hint poisons multi-turn, and the format matters as much
   as the presence of a hint.** I ran four hint formats plus no-hint on
   the same 138 failed tasks, paired:

   | format | what it is | recovery |
   |---|---|---:|
   | post | full block, appended *after* prompt as fake `solution_v1` | 0.0% [0.0, 2.7] |
   | current | full block, prepended as comment above function | 1.4% [0.4, 5.1] |
   | traceback | one line: `# Failed test: <assertion>` | 4.3% [2.0, 9.2] |
   | minimal | one line: `# Previous attempt was wrong. Try again.` | 5.1% [2.5, 10.1] |
   | **no hint** | resample only, no prompt change | **8.0% [4.5, 13.7]** |

   Paired McNemar across the family of four format-vs-no-hint
   contrasts, Holm-Bonferroni corrected: post p_holm=0.004 \*\*,
   current p_holm=0.035 \*, minimal/traceback NS. So it isn't
   "hints don't work"; it's that long structured hints that mimic
   code-comment blocks are significantly worse than no hint after
   correcting for multiple comparisons, and one-line
   natural-language hints are about a wash. Estimand is conditional
   recovery on the 138 single-turn failures, T=0.6 retries, paired
   by task. Replicable from `scripts/analyze_hint_sweep.py`.
4. **The retry hint also breaks correct solutions.** Take the 26
   tasks the model already passes, fabricate a "previous attempt was
   wrong" hint, regenerate. 8/26 = 30.8% break. Greedy decoding,
   deterministic, the hint is the only thing that changed.
5. **At a fixed 3-generation budget, multi-turn-no-hint beats pass@k.**
   pass@3 sampled = 18.8%, multi-turn-no-hint = 22.6%. The greedy
   turn 0 captures a high-mass mode that pure sampling at T=0.6
   misses. So the right policy isn't "more samples" or "more retries";
   it's greedy first, then sample on retry, with no hint.

Reading the failures gives the mechanism. Categorising the final-turn
output of all 138 retried tasks:

| format | all-comments | gave-up (`pass`) | wrote real code |
|---|---:|---:|---:|
| post | 36 | 59 | 43 |
| current | 12 | 25 | 101 |
| traceback | 10 | 35 | 93 |
| minimal | 10 | 25 | 103 |
| no hint | 3 | 28 | 107 |

The "post" format collapses 95/138 = 69% of retries into pure
boilerplate. The reason is in the prompt structure: by appending
`# def solution_v1: <old code> ... # try again below` *after* the
prompt, the model reads it as "the function is already finished,
write something else", and produces empty bodies or comment-only
outputs. The "current" format is gentler but still 4× the
all-comments rate of no-hint. HumanEval/41 is the cleanest example:
with the current hint, the model's final-turn output is literally
`# return 0`, then `# Failed test: assert candidate(0) == 0`, then
`# Fix the bug and try again.`, repeated. With no hint on the same
task at the same seed, it tries an actual `math.floor` expression.
The hint teaches a meta-pattern of commented-out attempt lines, and
on retry the model continues that pattern instead of re-entering
code.

The plot, briefly:

![recovery vs regression](assets/recovery_vs_regression.png)

Both with-hint Mellum points sit far below the y=x diagonal: their
recovery is dwarfed by their regression rate. DeepSeek-Coder-base
and DeepSeek-Coder-instruct sit *above* the diagonal in the
lower-left, with DS-instruct the highest of any with-hint point at
(4.4%, 18.9%). Mellum-SFT with no hint sits in the upper-left
because there is no hint to break correct solutions. The relative
positions are the comparison I'd put on a slide: "with the same
hint, what's the trade you're getting?" — and the answer changes
qualitatively across the four post-trainings.

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

## Cross-family check on DeepSeek-Coder-1.3B

If the hint-poisoning story is "code-completion models read commented
hints as code to honor", a different code-completion model trained on
broader data should show the same direction but a smaller magnitude
(more diverse training distribution → less brittle to atypical
comment blocks). Ran the same battery on
DeepSeek-Coder-1.3B-base (a code-completion model from a different
family, no instruction tuning):

| | single-turn pass@1 | multi-turn (hint) | multi-turn (no hint) | recovery (hint) | regression |
|---|---:|---:|---:|---:|---:|
| Mellum-4b-sft-python | 15.9% | 17.1% | 22.6% | 1.4% [0.4, 5.1] | **30.8% [16.5, 50.0]** |
| DeepSeek-Coder-1.3B-base | 25.0% | 28.0% | 37.2% | 4.1% [1.8, 9.3] | 2.4% [0.4, 12.6] |
| DeepSeek-Coder-1.3B-instruct | 54.9% | 63.4% | 67.1% | 18.9% [11.6, 29.3] | 4.4% [1.7, 10.9] |

Same hint format, same retry budget, same temperature. Three
contrasts worth noticing:

1. **Sign of the hint effect.** On Mellum-SFT, multi-turn-with-hint
   (17.1%) is *below* single-turn (15.9% on plus / 18.3% on base);
   on DeepSeek, multi-turn-with-hint (28.0%) is *above* single-turn
   (25.0%). Same prompt, opposite sign vs the model's own baseline.
2. **Regression rate.** 30.8% on Mellum vs 2.4% on DeepSeek. The
   95% CIs don't overlap. Mellum-SFT is uniquely brittle to this
   hint format; the effect is not generic to small code-completion
   models.
3. **Capability ceiling without the hint.** No-hint multi-turn on
   DeepSeek reaches 37.2% — a +12.2 pp lift over single-turn,
   compared with Mellum-SFT's +6.7 pp from 15.9% to 22.6%. So
   DeepSeek isn't just less hint-sensitive, its no-hint sampling
   is also doing more work. The hint is paying for itself less on
   DeepSeek (loses 9.2 pp vs no-hint) than on Mellum (loses 5.5 pp),
   but the starting point is high enough that the with-hint number
   is still net-positive vs single-turn. On Mellum the with-hint
   loss puts you below where you started.

I had read the original Mellum-only result as "any 1-4B code model
will be hint-poisoned." The DeepSeek run rejects that. The right
reading is closer to: Mellum's SFT-on-Python corpus probably
contains very few examples of comment blocks above functions that
look like retry hints, so they're out-of-distribution and the model
handles them by literal copying. DeepSeek's broader training mixes
in enough such patterns that it treats them more like context.

DeepSeek-Coder-1.3B-instruct (the chat-tuned sister, same
pre-training) closes the 2×2:

| | recovery (hint) | regression | net change at fixed budget |
|---|---:|---:|---:|
| Mellum-SFT | 1.4% | 30.8% | hint costs 5.5 pp vs no-hint |
| Mellum-DPO | 0.7% | 26.7% | – |
| DS-Coder-base | 4.1% | 2.4% | hint costs 9.2 pp vs no-hint |
| DS-Coder-instruct | **18.9%** | 4.4% | hint costs 3.7 pp vs no-hint |

Three things this pins down. Cross-family regression-rate
comparisons use Fisher's two-sided exact test on the 2×2 counts
(rather than just CI overlap, since the exposure sets differ —
each model's regression denominator is the tasks *that model*
passes single-turn).

- **Mellum-{SFT, DPO} false-negative retry sensitivity is materially
  higher than DeepSeek's, in this prompt and benchmark.** Confirmatory
  family is the four cross-family Mellum-vs-DeepSeek contrasts,
  Holm-Bonferroni corrected: Mellum-SFT vs DS-instruct p_holm=0.003
  \*\*, Mellum-SFT vs DS-base p_holm=0.005 \*\*, Mellum-DPO vs
  DS-instruct p_holm=0.028 \*, Mellum-DPO vs DS-base p_holm=0.028 \*.
  All four cross-family contrasts significant after correction.
  Exploratory within-family rates are statistically indistinguishable
  (Mellum-SFT vs DPO p=1.0; DS-base vs instruct p=1.0). The hint
  poisoning is consistent across Mellum's two post-trainings and
  absent in both of DeepSeek's. The causal "Mellum's training
  distribution" reading is consistent with the data but the
  experiment doesn't directly identify it — see "What I'd do next"
  for the shared-task canonical-solution assay that would.
- **Recovery scales with capability, but the no-hint policy still
  wins for every model.** Going Mellum-SFT → DS-base → DS-instruct,
  hint-recovery rises 1.4% → 4.1% → 18.9%, but no-hint multi-turn
  beats with-hint multi-turn at the same retry budget for all four
  models. The hint *is* useful enough on DS-instruct that adding
  it costs only 3.7 pp vs no-hint, but it never wins.
- **The published "multi-turn pass@1" number conflates capability
  with hint-format compatibility.** A leaderboard that ranks models
  by multi-turn pass@1 with one hardcoded hint format will rank
  DS-instruct (63.4%) above DS-base (28.0%) above Mellum-SFT
  (17.1%) — but the gap between Mellum-SFT and the rest is
  partly that Mellum reads this hint as code-to-honor while the
  others read it as instructions-to-follow. With a different hint
  the ordering between the closely-spaced ones could swap, even
  though capability didn't change.

## What "regression rate" actually measures

A note on framing, because I had it slightly wrong on first writeup. The
regression test fabricates a "previous attempt was wrong" hint on tasks
the model already passes and reruns the retry pipeline. What it measures
is **false-negative retry sensitivity**: the probability that a retry
trigger fired in error damages a correct answer. It is not "the cost the
average user pays per retry." That number is `regression_rate ×
P(retry-triggered-when-correct)`, and the second factor is deployment-
specific (an IDE that auto-retries on every TODO comment has a high
trigger rate; a user clicking Retry has a low one).

Even with that caveat, 30.8% [16.5%, 50.0% Wilson] is large. It says
that any mechanism that retries without high confidence the previous
attempt was wrong, on Mellum-SFT, throws away one in three correct
answers it touches. The minimum reportable pair for any "multi-turn
helps" claim should be the recovery rate **paired with** the false-
negative retry sensitivity, plus an estimate of how often the deployed
retry trigger fires on correct outputs.

The published multi-turn pass@1 number reports only the first of those
three. That's the part of the framing this project is trying to make
explicit.

A specific caveat on the comparison between recovery (1.4%) and
regression (30.8%): they aren't the same procedure, so they aren't
the same axis. Recovery uses real failure feedback as the hint,
sampled retries at T=0.6, up to two attempts, on the 138
single-turn failures. Regression uses a static fabricated hint
(`expected output X, got Y`), one greedy regenerate, on the 26
already-passing tasks. Both are answering "what does this hint do
to the model under deployment-shape conditions," but the recovery
number is the model's best chance under realistic retry pressure,
and the regression number is a conservative stress test under one
fabricated trigger.

### The no-hint regression control narrows the claim

`scripts/run_regression_nohint.py` runs the regression test with
the hint stripped — same passing tasks, same retry budget, same
T=0.6 sampling, no fabricated hint. The point is to isolate
"hint-induced damage" from "stochastic-retry damage on a from-scratch
problem". Numbers on the same passing-task denominator as each
model's with-hint regression:

| | with current hint | no-hint control | hint marginal |
|---|---:|---:|---:|
| Mellum-SFT (n=26) | 30.8% | 26.9% | +3.8 pp |
| Mellum-DPO (n=15) | 26.7% | 33.3% | -6.7 pp |
| DS-Coder-base (n=41) | 2.4% | 22.0% | -19.6 pp |
| DS-Coder-instruct (n=90) | 4.4% | 22.2% | -17.8 pp |

So most of what looked like hint-induced damage is just stochastic
retry damage on a from-scratch problem (sampled retry breaks
22-33% of correct answers in three of four models even with no
hint). The strongest defensible reading: on the two DeepSeek
models the hint reduces sampled-retry breakage substantially; on
Mellum-SFT it doesn't. The "stabilising" effect on DeepSeek has
an alternative mechanism — the verbose hint injects the prior
correct completion back into the prompt, so the model may be
copying a correct program it is directly shown rather than
exhibiting general robustness — and the within-Mellum sign
difference (+3.8 vs -6.7 pp) rests on n=15 and n=26 where the
discordant-task counts are 5 vs 4 and 3 vs 4 respectively, so
the per-Mellum sign isn't statistically meaningful. Two
confounds I haven't fully addressed: (a) the with-hint regression
uses greedy decoding while the no-hint control uses T=0.6
sampled, so this comparison still confounds prompt content with
decoding mode; (b) the canonical-poisoning experiment below
shows DeepSeek-base lifts +32 pp from seeing the canonical, so
"hint as anchor" is the more conservative reading than "hint
stabilises sampled retries."

A cleaner settling experiment: full 2×2 of {hint, no-hint} ×
{greedy, T=0.6} on the same passing tasks, plus a fifth arm with
the failure framing but the prior code stripped. That separates
"anchor from prior completion" from "anchor from failure framing."
Not run here.

### Canonical-solution poisoning: shared-task assay

Each model's regression denominator is different (Mellum-SFT
n=26, DS-instruct n=90), which is the right thing to ask in
deployment but conflates "models pass different tasks" with "models
react to hints differently". The shared-task version: give every
model the *same* canonical HumanEval solution as the "previous
attempt" plus the wrong-hint, run the same greedy regeneration on
all 164 tasks. Pass-after vs natural single-turn pass@1:

| | natural | with canonical context | lift |
|---|---:|---:|---:|
| Mellum-DPO | 9.1% | 12.8% | +3.7 pp |
| Mellum-SFT | 15.9% | 31.1% | +15.2 pp |
| DS-Coder-instruct | 54.9% | 72.6% | +17.7 pp |
| DS-Coder-base | 25.0% | 57.3% | **+32.3 pp** |

DS-base shows the largest lift: 25 → 57 = +32 pp pass rate
when given the canonical solution above the prompt and the
"wrong, try again" flag. Mellum-DPO shows the smallest: 9 → 13 =
+3.7 pp. The pattern is consistent with three different mechanisms
that this experiment can't disentangle from pass/fail alone:
(a) imitation propensity (the model copies the canonical it just
saw); (b) correctness recognition (the model looks at the canonical,
recognises it as right, ignores the wrong-flag); (c) differential
capacity (DS-base has more headroom, larger absolute swings
expected from any kind of context shift). The lift table is real
but the "DPO steered Mellum away from imitation" reading is one
hypothesis among three, not a measurement.

A cleaner settling experiment would store completions and compute
exact-match / AST-similarity rate to the provided canonical, and
add control arms with (a) an incorrect canonical-like distractor,
(b) a natural-language description with no code, (c) a
variable-renamed paraphrase of the canonical. Not run here.

## What's in here

```
eval/sandbox.py     subprocess + 10s timeout, returns pass/fail and a
                    short feedback string. -I flag on the interpreter
                    because /tmp/inspect.py once ate two hours.
eval/loaders.py     joins openai/openai_humaneval and evalplus/humanevalplus
                    on task_id so one completion is scored under both
                    test suites
eval/runner.py      AutoModel wrapper with completion + FIM-mode
                    (model-aware: Mellum's <fim_prefix>/<fim_suffix>/
                    <fim_middle> + filename, DeepSeek's full-width
                    pipe variant), four retry-hint formats, and a
                    tokenizer-bug fallback for DeepSeek-Coder under
                    transformers 5.7
eval/multi_turn.py  retry loop with use_hint + hint_format selector
eval/loaders.py     joins openai_humaneval and evalplus/humanevalplus
                    on task_id so one completion is scored under both
eval/fim_loaders.py loads OpenAI HumanEval-Infilling (single-line,
                    multi-line, random-span, light)
eval/mbpp_loader.py MBPP+ via evalplus/mbppplus
eval/report.py      cross-table with Wilson CIs + codex-style pass@k
scripts/run_*.py    one runner per experiment, all take MODEL_PATH +
                    TAG positional args (singleturn, multiturn,
                    multiturn_nohint, regression, regression_nohint,
                    canonical_poisoning, passk, fim, hint_sweep,
                    humaneval_infilling, mbpp_singleturn)
scripts/analyze_hint_sweep.py   paired McNemar + Holm-Bonferroni
scripts/summary.py              one-command audit of every README claim
scripts/calibrate_temperature.py 8-task slice for picking T
scripts/taxonomy.py             classify final-turn outputs by mode
scripts/make_plot.py            recovery vs regression scatter
data/humaneval-infilling/       four FIM variants from openai/human-eval-infilling
results/                        JSONL per (tag, experiment), one row per task
```

About 1300 lines of Python excluding smoke / inspection helpers.

## Things that didn't work, and one thing I want to flag

- First multi-turn run produced byte-identical output on every retry.
  Greedy decoding plus a small comment-prefixed prompt rarely flips
  the argmax, so the "retry" was a no-op. Switched to temperature 0.6
  with a per-task seed for retries only. Turn 0 stays greedy so the
  single-turn number matches what a leaderboard would post.
- Halfway through, every plus test started failing with
  `AttributeError: module 'inspect' has no attribute 'cleandoc'`. Took
  half a day to find and a one-line fix to repair. Full writeup in
  [`post-mortem.md`](./post-mortem.md). Short version: I'd dropped a
  `/tmp/inspect.py` while debugging, Python prepends a subprocess
  script's directory to `sys.path`, my five-line shadow file beat
  stdlib `inspect`, numpy then fails to import. `python -I` fixes it.
- I expected Mellum-SFT to outperform Mellum-base on HumanEval. The
  opposite ordering (base > SFT > DPO) is the writeup.
- I expected the retry hint to help recovery. The opposite is true on
  Mellum (and the "post" format is catastrophic, 0/138 recoveries).
  On DeepSeek-Coder-base the hint helps, +3 pp.
- DeepSeek-Coder's tokenizer in `transformers` 5.7.0 silently strips
  whitespace from BPE encodings: `def add(a, b)` round-trips to
  `defadd(a,b)`. First DeepSeek-base run produced 0/164 pass@1 because
  the model was being shown text-without-spaces and emitted Python that
  was syntactically broken. The fix is to bypass `AutoTokenizer` and
  load `tokenizer.json` directly via `tokenizers.Tokenizer.from_file`,
  wrapped in `PreTrainedTokenizerFast`. `eval/runner.py` now sniffs
  for the bug (round-trip a sentinel) and falls back if it fires.

## What I'd do next if I had another week

- **FIM-aware test mutation.** EvalPlus's mutation strategy doesn't
  transfer to FIM benchmarks. The right shape is to mutate the
  *surrounding file context* (rename variables, reorder imports, add
  an unrelated function above), AST-level, and re-evaluate against
  SAFIM or RepoBench. That gives a robustness number in the
  deployment-relevant direction (which is what JetBrains actually
  ships against, vs HumanEval).
- **Five-format sweep on stronger models.** Mellum's hint poisoning
  is most visible because Mellum's training distribution makes these
  comment blocks unfamiliar. On the four models tested here the
  no-hint policy wins, but the *gap* shrinks substantially with
  capability (5.5 pp on Mellum, 3.7 pp on DS-instruct). Worth
  running the same paired-McNemar five-format sweep on Qwen2.5-
  Coder-Instruct and StarCoder2-7B to find out whether the gap
  closes or flips signs at higher capability.
- **Calibrate the regression-rate trigger.** Regression rate is
  `P(correct → broken | retry triggered)`. The deployed cost is
  that times `P(retry triggered | correct)`. The latter is
  deployment-specific (an IDE that auto-retries on every TODO
  comment is high; a human clicking "regenerate" is low). The
  natural follow-up is to instrument a real IDE retry trigger
  on a Mellum deployment and measure the conditional probability
  in the wild.

## Reproducing

Runs were on a 4090 (24GB). Wallclocks for the full Mellum-4b family:

```bash
uv sync
# Mellum-SFT primary battery
PYTHONPATH=. uv run python scripts/run_singleturn.py        $M/mellum-sft-python mellum_sft   # ~13 min
PYTHONPATH=. uv run python scripts/run_multiturn.py         $M/mellum-sft-python mellum_sft   # ~45 min
PYTHONPATH=. uv run python scripts/run_multiturn_nohint.py  $M/mellum-sft-python mellum_sft   # ~30 min
PYTHONPATH=. uv run python scripts/run_regression.py        $M/mellum-sft-python mellum_sft   # ~5 min
PYTHONPATH=. uv run python scripts/run_passk.py             $M/mellum-sft-python mellum_sft   # ~50 min, 4 samples T=0.6
PYTHONPATH=. uv run python scripts/run_hint_sweep.py        $M/mellum-sft-python mellum_sft   # ~90 min, 3 formats
# Mellum-base + Mellum-DPO
PYTHONPATH=. uv run python scripts/run_singleturn.py        $M/mellum-base       mellum_base  # ~18 min
PYTHONPATH=. uv run python scripts/run_fim.py               $M/mellum-base       mellum_base  # ~5 min
# DeepSeek cross-family
PYTHONPATH=. uv run python scripts/run_singleturn.py        $M/deepseek-coder-1.3b-base ds_base  # ~10 min
PYTHONPATH=. uv run python scripts/run_multiturn.py         $M/deepseek-coder-1.3b-base ds_base  # ~30 min
PYTHONPATH=. uv run python scripts/run_regression.py        $M/deepseek-coder-1.3b-base ds_base  # ~3 min
# Reports + plots
PYTHONPATH=. uv run python -m eval.report
PYTHONPATH=. uv run python scripts/analyze_hint_sweep.py
PYTHONPATH=. uv run python scripts/taxonomy.py
PYTHONPATH=. uv run python scripts/make_plot.py
```

The torch index in `pyproject.toml` is pinned to cu128 so the lockfile
resolves on a fresh CUDA box. To run on Apple silicon, drop the
`[tool.uv.sources]` block and let uv resolve a CPU build.

## Things this is not

This is a small probe, not a benchmark. Two model families, one
benchmark family, one decoding configuration. The conclusions worth
defending from this much data are "the way pass@1 with retries is
reported needs more components than it currently has" and "Mellum's
HumanEval scores under-represent the model in two specific ways" and
"the regression rate effect is uniquely large on Mellum-SFT, not
generic to small code-completion models." Stronger versions of any of
those would require at least one chat-tuned family and a wider hint
sweep across more models.

Underlying paper that started this:
[Liu et al., NeurIPS 2023, "Is Your Code Generated by ChatGPT Really
Correct?"](https://arxiv.org/abs/2305.01210). Mellum models from
JetBrains are on [HuggingFace](https://huggingface.co/JetBrains).
