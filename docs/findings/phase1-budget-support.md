# Phase 1 — Does gepa v0.1.4 already support a dollar budget?

Date: 2026-08-07 · Scope: read-only source audit of the pinned baseline release.

## Verdict

**Partially — and not in the way we need.** v0.1.4 ships a USD-denominated stopper,
but it meters the **reflection LM only**. Total spend (solver/task LM + reflection)
has no built-in stop. For a SWE-Bench solver→refiner program the task LM dominates
cost by a wide margin, so the shipped stopper does **not** satisfy hard rule 2's
"dollar-budget stop criterion" on its own.

We still need to build a component — but it is smaller than expected, because the
extension point we need already exists and is public.

## What already exists

`src/gepa/utils/stop_condition.py` defines a `StopperProtocol` — any callable
`(GEPAState) -> bool` — plus these implementations:

| Stopper | Meters |
|---|---|
| `MaxMetricCallsStopper` | `state.total_num_evals` |
| `TimeoutStopCondition` | wall clock |
| `FileStopper` | presence of a stop file |
| `SignalStopper` | SIGINT/SIGTERM |
| `ScoreThresholdStopper` | best val score |
| `NoImprovementStopper` | iterations without improvement |
| `MaxTrackedCandidatesStopper` | `len(state.program_candidates)` |
| `MaxCandidateProposalsStopper` | `state.i` |
| **`MaxReflectionCostStopper`** | **`reflection_lm.total_cost` (USD)** |
| `CompositeStopper` | combines the above, `mode="any"` / `"all"` |

`gepa.optimize()` accepts `stop_callbacks: StopperProtocol | Sequence[StopperProtocol]`
and a convenience `max_reflection_cost: float` (api.py:72–74). Multiple stoppers are
folded into a `CompositeStopper` with `mode="any"` (api.py:284–295). At least one of
`stop_callbacks`, `max_metric_calls`, or `max_reflection_cost` is required.

Cost accounting lives in `gepa.lm.LM`, which wraps `litellm.completion` and
accumulates `litellm.completion_cost(...)` into `_total_cost`, exposed as a
`total_cost` property (lm.py:117, 127, 74). A plain callable passed as
`reflection_lm` is wrapped in `TrackingLM`, which **always reports
`total_cost = 0.0`** — so a cost stopper over a custom callable silently never fires.
api.py:268–278 guards the `reflection_strategy` variant of this with an explicit
error, and the docstring on `MaxReflectionCostStopper` calls the `TrackingLM` case out.

**Gap:** the task/solver LM is invoked inside the *adapter*, which gepa does not own
and does not meter. Nothing in v0.1.4 aggregates solver spend, so nothing can stop on it.

## Behaviour-neutrality: verified

Hard rule 2 requires that our stop component observe spend and decide when to stop,
touching nothing else. Auditing the engine confirms a custom stopper can meet this:

1. **The stopper is consulted at exactly one place.** `_should_stop()` (engine.py:994)
   has exactly one call site: the main loop condition `while not self._should_stop(state)`
   (engine.py:731). It is a pure loop-exit check evaluated between iterations — it cannot
   influence candidate selection, reflection, sampling, merge scheduling, or acceptance.

2. **`GEPAState` is passed by reference but only read.** All shipped stoppers read
   state; ours will too. This is a convention we must respect, not something the
   framework enforces — noted as a review item for our component.

3. **⚠️ One duck-typed leak to avoid.** The engine introspects the stopper for a
   `max_metric_calls` attribute in two places:
   - `_get_remaining_budget()` (engine.py:1002–1020), unwrapping `CompositeStopper`
     via its `.stoppers` attribute;
   - the tqdm total in `run()` (engine.py:564–583).

   Both consumers are **reporting-only**: `_get_remaining_budget()` feeds solely the
   `metric_calls_remaining` field of the `BudgetUpdatedEvent` callback (engine.py:719),
   and the other sets a progress-bar total. Neither feeds optimization logic — verified
   by grepping every reference to `_get_remaining_budget` / `remaining_budget` across
   `core/`, `strategies/`, and `proposer/`; those two are the only ones.

   **Consequence:** our dollar stopper must *not* expose an attribute named
   `max_metric_calls`. If it does, it would hijack the progress bar and the
   budget-event field. Exposing anything else is inert. This is the single concrete
   constraint on the design, and it is cheap to honour and to test.

Conclusion: a stopper is genuinely a behaviour-neutral extension point in v0.1.4.
The baseline runs can use one without violating baseline purity.

## What we need to build

A `MaxTotalCostStopper` that meters **solver + reflection** spend in USD.

Two candidate metering sources, both already proven in-repo:

- **A — litellm global success callback.** `src/gepa/gskill/gskill/cost_tracker.py`
  already does exactly this: registers `litellm.success_callback` and accumulates
  `litellm.completion_cost(...)` thread-safely across *every* call, agent and
  reflection alike. Catches all spend regardless of who issues the call.
  Caveat: it *assigns* `litellm.success_callback = [self._on_completion]`, clobbering
  any other callbacks — ours must append, not overwrite. Its reflection/agent split
  is also heuristic (`"pro" in model.lower()`), which we would not reuse.
- **B — explicit LM handles.** Sum `total_cost` over an explicit list of `gepa.lm.LM`
  instances (solver LM + reflection LM). Precise and dependency-free, but only works
  if the adapter routes all its calls through `LM` objects we hold — which is a
  constraint on our adapter design, not a given.

Recommendation: **B as the primary, A as a fallback/cross-check.** B is explicit and
testable without touching litellm global state; A is the safety net for spend that
leaks around the LM handles, and doubles as an independent audit of B's number. Since
we control the solver→refiner adapter, B's precondition is ours to guarantee.

Then compose:

```python
gepa.optimize(..., stop_callbacks=[MaxTotalCostStopper(budget_usd, lms=[solver_lm, reflection_lm])])
```

`CompositeStopper(mode="any")` lets us add a wall-clock or metric-call safety net
alongside it without further code.

### Granularity caveat (must be surfaced in the writeup)

The stopper fires **between iterations**, so the realised spend overshoots the budget
by up to one iteration's cost. On SWE-Bench an iteration is expensive, so the
overshoot is material and varies per seed. Options: (a) accept and report actual
spend per seed, (b) set the threshold at a fraction of the nominal budget, (c) also
check inside the adapter's rollout loop — but (c) is *not* behaviour-neutral and must
not be used for baselines. Recommend (a): report actual spend, since seeds must be
compared on equal *realised* cost anyway.

## Upstream PR value

`MaxTotalCostStopper` is a clean, self-contained addition to
`gepa/utils/stop_condition.py` that fills a real gap the maintainers have already
started on (they shipped the reflection-only version). It fits their conventions
exactly. Good PR candidate.

Note: `MaxReflectionCostStopper` is importable from `gepa.utils` and settable via the
`max_reflection_cost=` API kwarg, but is **omitted from the `gepa/__init__.py`
`__all__`-style export list** that names the other eight stoppers — a plausible
oversight worth a one-line fix in the same PR.
