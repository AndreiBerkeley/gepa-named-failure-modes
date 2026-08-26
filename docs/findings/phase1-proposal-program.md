# Phase 1 — PROPOSAL: solver→refiner candidate program for SWE-Bench

Date: 2026-08-07 · Status: **awaiting Andrei's decision**. Nothing built yet.

## What GEPA is optimizing

Two text components, both evolved by GEPA's reflective mutation:

- `solver_instruction` — how to turn a problem statement + retrieved code into a patch.
- `refiner_instruction` — how to diagnose and repair a candidate patch given feedback.

Everything else (retrieval, patch application, grading) is fixed scaffolding. Keeping
the optimizable surface to exactly two instructions is what makes the baseline-vs-taxonomy
comparison interpretable: the only thing that differs between arms is how those two
strings are refined.

## Proposed pipeline

```
instance (problem_statement, repo@base_commit)
  │
  ├─ [fixed] retrieve context: BM25 over repo files at base_commit → top-k files
  │
  ├─ SOLVER   (LM, prompt = solver_instruction)      → unified diff
  │
  ├─ [fixed] apply patch → structured feedback:
  │            • does it apply cleanly?
  │            • does the repo still import / parse?
  │            • (option) existing repo test subset result
  │
  ├─ REFINER  (LM, prompt = refiner_instruction)     → revised unified diff
  │            input: problem_statement, retrieved context, candidate patch, feedback
  │
  └─ [fixed] grade: SWE-bench harness → FAIL_TO_PASS ∧ PASS_TO_PASS → score ∈ {0,1}
```

Two LM calls per rollout, fixed. No variable-length agent loop.

## The three decisions inside this design

### 1. How the solver sees the repo — **recommend BM25 retrieval**

| Option | Trade-off |
|---|---|
| **BM25 retrieval (recommend)** | Deterministic, fixed cost, ~2 LM calls/rollout. SWE-bench ships BM25 retrieval tooling (`swebench.inference.make_datasets.bm25_retrieval`) and pre-built retrieval datasets, so this is well-trodden. Ceiling on solve rate is lower than an agent's. |
| Oracle file list (use `patch` to reveal edited files) | Much higher solve rate, but **leaks the answer's location** — inflates scores and makes the failure taxonomy describe a task nobody actually faces. Reject. |
| Agentic loop (mini-swe-agent style) | Highest solve rate and closest to SOTA practice. But cost per rollout is unbounded and high-variance, which fights the dollar-budget stopper, and variable-length traces make failure-mode taxonomy generation much noisier. |

BM25 is recommended chiefly because **cost predictability and trace comparability are
load-bearing for this specific study**. A variable-cost agent would make the three
baseline seeds non-comparable under a fixed dollar budget (one seed might get 40
iterations, another 12), which would confound the very comparison we are running.

Worth stating plainly: this caps absolute solve rate well below published SOTA. That
is acceptable — we are measuring a *delta* between two GEPA variants on an identical
scaffold, not competing on the leaderboard. The blog post must not present these
numbers as SWE-Bench SOTA comparisons.

### 2. What feedback the refiner sees — **recommend "cheap signals only"**

This is the design's biggest integrity risk.

| Feedback | Verdict |
|---|---|
| Patch applies cleanly / rejects | ✅ Free, no leakage. |
| Syntax + import check on modified files | ✅ Cheap, no leakage. |
| Repo's **own existing** tests (a bounded subset) | ⚠️ Legitimate and standard, but costs a container run per rollout and roughly doubles eval time. |
| **`FAIL_TO_PASS` test output** | ❌ **Reject.** That is the grading signal. Feeding it to the refiner at inference time is training on the test — it would make results meaningless. |

Recommend applies-cleanly + syntax/import only for the main runs. It keeps every
rollout to two LM calls plus one cheap static check, and keeps grading honest.

`PASS_TO_PASS` is likewise part of grading and must not be surfaced to the refiner.

### 3. Adapter shape — **custom `GEPAAdapter`, not DSPy**

gepa's `GEPAAdapter` protocol (`gepa.core.adapter`) wants `evaluate(...)` returning an
`EvaluationBatch` plus reflective dataset construction. Implementing it directly (rather
than going through the DSPy adapter) gives us control over exactly what lands in the
reflective dataset — which is precisely the surface Phase 5 will condition on the
taxonomy. Going through DSPy would put an abstraction between us and that surface.

This also makes the adapter a clean PR candidate for upstream: gepa has a
`terminal_bench_adapter` but no SWE-Bench adapter.

**Trace capture is designed in from the start, not bolted on in Phase 3:** every
rollout emits a structured record (retrieved files, solver patch, feedback, refiner
patch, grade, per-stage token/cost). Phase 3's harvest then becomes a filter over
already-captured data, and Phase 4's taxonomy sees uniform, comparable traces.

## Cost per rollout (rough, for budget setting)

| Item | Estimate |
|---|---|
| Solver call (~15–25k in, ~1–2k out) | dominant LM cost |
| Refiner call (similar) | ~1× solver |
| Grading container run | compute, not tokens |

So ≈2 solver-equivalent LM calls per rollout. Multiply by
`n_val × n_candidates + minibatch rollouts + n_test` from the split proposal. I have
deliberately not put dollar figures here — they depend entirely on the model choice,
which is your deferred decision. Once you pick a solver model I can produce a concrete
per-seed budget estimate.

## Blocked on you

1. **Retrieval strategy** — BM25 (recommended), or accept agentic cost variance?
2. **Refiner feedback level** — cheap static signals (recommended), or add a bounded
   existing-test run?
3. **Solver / refiner model choice** — deferred per CLAUDE.md; needed before I can
   cost anything. Same model for both stages, or a cheaper refiner?
4. Confirm the **custom `GEPAAdapter`** route over DSPy.

## Deliberately not decided here

Per CLAUDE.md, these remain yours: budget per seed, model choice, and the
taxonomy-integration mechanism (Phase 5). This proposal is scoped to the *baseline*
program shape only — it does not presuppose how the taxonomy will later be injected,
beyond ensuring the reflective dataset is a surface we control.
